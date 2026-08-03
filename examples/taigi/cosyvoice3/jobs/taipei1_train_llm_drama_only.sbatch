#!/usr/bin/env bash
# 消融實驗：只用戲劇資料訓練 LLM，排除 naer 朗讀語料。
#
# naer 是朗讀語料，風格與八點檔對白差距大。它補了新聞語域的詞彙，
# 但也可能稀釋口語風格。這支 job 自己從既有的 Parquet 清單濾掉 naer，
# 不需要重跑一次資料準備（省下約 5 小時）。
#
# 過濾在 job 內完成，所以可以和訓練一起用 --dependency 預先送出，
# 不必等資料準備結束後再由人介入。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-llm-drama
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"

tag="${TP1_TAG:-_v3d_drama}"
tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/llm_ddp${tag}" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/llm_ddp${tag}" \
  "${tp1_cache}"

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${tp1_cache}:/tmp/cosyvoice-cache"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c "
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /models/Fun-CosyVoice3-0.5B/llm.pt
    test -s /workspace/prepared/data/train/parquet/data.list

    # Parquet 是照 utterance ID 排序打包的，而 ID 帶來源前綴（drama1- / drama2- /
    # naer-），所以幾乎每個檔案都是單一來源，只有交界處會混。這裡只保留 100%
    # 戲劇的檔案，混到 naer 的整個丟掉——寧可少幾千句，也不要讓消融實驗不乾淨。
    python - <<'PY'
import pyarrow.parquet as pq
from pathlib import Path

for split in ('train', 'dev'):
    src = Path(f'/workspace/prepared/data/{split}/parquet/data.list')
    kept, dropped, mixed = [], 0, 0
    for line in src.read_text().split():
        utts = pq.read_table(line, columns=['utt'])['utt'].to_pylist()
        drama = sum(1 for u in utts if u.startswith(('drama1-', 'drama2-')))
        if drama == len(utts):
            kept.append(line)
        else:
            dropped += 1
            if drama:
                mixed += 1
    out = src.with_name('data_drama_only.list')
    out.write_text('\n'.join(kept) + '\n')
    print(f'{split}: 保留 {len(kept)} 個 parquet，丟棄 {dropped}（其中 {mixed} 個是混合檔）', flush=True)
PY

    test -s /workspace/prepared/data/train/parquet/data_drama_only.list
    test -s /workspace/prepared/data/dev/parquet/data_drama_only.list
    python -c 'import torch; assert torch.cuda.device_count() == 8, torch.cuda.device_count()'

    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config \"\${TP1_CONFIG:-/workspace/conf/taigi_cosyvoice3_v1.yaml}\" \
      --train_data /workspace/prepared/data/train/parquet/data_drama_only.list \
      --cv_data /workspace/prepared/data/dev/parquet/data_drama_only.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /models/Fun-CosyVoice3-0.5B/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp${tag} \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp${tag} \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  "
