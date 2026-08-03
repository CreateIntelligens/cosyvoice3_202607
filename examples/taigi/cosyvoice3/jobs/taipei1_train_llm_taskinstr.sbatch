#!/usr/bin/env bash
# 實驗：讓 instruction 誠實描述輸入的文字系統。
#
# 兩份語料其實在教兩個不同的任務：戲劇字幕是華語漢字（華語 → 台語音），
# naer 是台文漢字與漢羅混寫（台文 → 台語音，不需翻譯）。實測特徵字頻率：
# 「毋」naer 5,317 次 / drama 1 次，「佇」5,219 / 0，「袂」2,705 / 1。
#
# 但現行流程對兩者掛同一句「請將輸入的華語翻譯成臺灣台語」，這對 naer 的
# 94,081 句（訓練資料的 20%）是錯的描述。這支 job 給各自正確的 instruction，
# 讓 naer 貢獻發音與韻律，又不污染「華語翻台語」這個目標任務。
# 推論時固定用華語那一句。
#
# 只有 instruct 檔要重寫，wav.scp / text / utt2spk / embedding / speech token
# 全部沿用既有產物（符號連結），所以只需重跑 stage 3 打包 Parquet。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-llm-taskinstr
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"
: "${TP1_DATASETS:?Set TP1_DATASETS：materialize=none，打包 Parquet 時要讀原始 WAV}"

tag="${TP1_TAG:-_v3g_taskinstr}"
tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/llm_ddp${tag}" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/llm_ddp${tag}" \
  "${tp1_cache}"

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${TP1_DATASETS}:/datasets:ro,${tp1_cache}:/tmp/cosyvoice-cache"

# materialize=none 時 naer 的 wav.scp 指向 /naer_flat/...，打包 Parquet 要讀得到。
# 少掛這兩個會讓 make_parquet_list 在 naer 的第一個檔案就 FileNotFoundError。
: "${TP1_NAER:?Set TP1_NAER}"
: "${TP1_NAER_FLAT:?Set TP1_NAER_FLAT}"
tp1_mounts="${tp1_mounts},${TP1_NAER}:/naer:ro,${TP1_NAER_FLAT}:/naer_flat:ro"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c "
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /models/Fun-CosyVoice3-0.5B/llm.pt

    python - <<'PY'
from pathlib import Path
import os

# 戲劇是華語輸入，要翻譯；naer 是台文輸入，直接唸出來即可。
DRAMA = 'You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>'
NAER  = 'You are a helpful assistant. 請將輸入的臺灣台語文朗讀出來。<|endofprompt|>'

src_root = Path('/workspace/prepared/data')
dst_root = Path('/workspace/prepared_styletag/data')
for split in ('train', 'dev'):
    src, dst = src_root / split, dst_root / split
    dst.mkdir(parents=True, exist_ok=True)
    # 重的產物一律連結，不複製
    for name in ('wav.scp', 'text', 'utt2spk', 'utt2embedding.pt',
                 'spk2embedding.pt', 'utt2speech_token.pt'):
        s, d = src / name, dst / name
        if s.exists() and not d.exists():
            os.symlink(s, d)
    counts = {'drama': 0, 'naer': 0}
    lines = []
    for line in (src / 'instruct').read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        utt = line.split(maxsplit=1)[0]
        if utt.startswith('naer-'):
            counts['naer'] += 1
            lines.append(f'{utt} {NAER}')
        else:
            counts['drama'] += 1
            lines.append(f'{utt} {DRAMA}')
    (dst / 'instruct').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'{split}: 華語輸入 {counts[\"drama\"]:,} 句、台文輸入 {counts[\"naer\"]:,} 句', flush=True)
PY

    for split in train dev; do
      python tools/make_parquet_list.py \
        --num_utts_per_parquet 1000 --num_processes 16 \
        --src_dir /workspace/prepared_styletag/data/\${split} \
        --des_dir /workspace/prepared_styletag/data/\${split}/parquet
    done
    wc -l /workspace/prepared_styletag/data/*/parquet/data.list

    python -c 'import torch; assert torch.cuda.device_count() == 8, torch.cuda.device_count()'
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config \"\${TP1_CONFIG:-/workspace/conf/taigi_v3_ep10.yaml}\" \
      --train_data /workspace/prepared_styletag/data/train/parquet/data.list \
      --cv_data /workspace/prepared_styletag/data/dev/parquet/data.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /models/Fun-CosyVoice3-0.5B/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp${tag} \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp${tag} \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  "
