#!/usr/bin/env bash
# 消融實驗：只用 naer 朗讀語料訓練 LLM，排除戲劇資料。
#
# 與 taipei1_train_llm_drama_only.sbatch 互補。兩者加上用全部資料的實驗 A，
# 構成完整的消融：A = 戲劇 + 朗讀、D = 只有戲劇、E = 只有朗讀。
#
# 過濾在 job 內完成，不需要重跑資料準備。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-llm-naer
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"

tag="${TP1_TAG:-_v3e_naer}"
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

    # Parquet 照 utterance ID 排序打包，而 ID 帶來源前綴，所以幾乎每個檔案都是
    # 單一來源，只有交界處會混。只保留 100% naer 的檔案，混到戲劇的整個丟棄。
    python - <<'PY'
import pyarrow.parquet as pq
from pathlib import Path

for split in ('train', 'dev'):
    src = Path(f'/workspace/prepared/data/{split}/parquet/data.list')
    kept, dropped, mixed = [], 0, 0
    for line in src.read_text().split():
        utts = pq.read_table(line, columns=['utt'])['utt'].to_pylist()
        naer = sum(1 for u in utts if u.startswith('naer-'))
        if naer == len(utts):
            kept.append(line)
        else:
            dropped += 1
            if naer:
                mixed += 1
    out = src.with_name('data_naer_only.list')
    out.write_text('\n'.join(kept) + '\n')
    print(f'{split}: 保留 {len(kept)} 個 parquet，丟棄 {dropped}（其中 {mixed} 個是混合檔）', flush=True)
PY

    test -s /workspace/prepared/data/train/parquet/data_naer_only.list
    test -s /workspace/prepared/data/dev/parquet/data_naer_only.list
    python -c 'import torch; assert torch.cuda.device_count() == 8, torch.cuda.device_count()'

    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config \"\${TP1_CONFIG:-/workspace/conf/taigi_cosyvoice3_v1.yaml}\" \
      --train_data /workspace/prepared/data/train/parquet/data_naer_only.list \
      --cv_data /workspace/prepared/data/dev/parquet/data_naer_only.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /models/Fun-CosyVoice3-0.5B/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp${tag} \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp${tag} \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  "
