#!/usr/bin/env bash
# 概念驗證：從既有的台語模型出發，用 SuíSiann 教它讀台羅數字調。
#
# 要回答的是是非題，不是要訓成品。4.62 小時、單一語者，成品不可能。問題是：
# 給模型「khian5-khun1」，它會不會念出 khiân-khun。會的話，未見字就從
# 「補訓練資料」變成「查辭典」，因為台語連調音節只有 2,269 個而漢字沒有上限。
#
# 從 TP1_INIT_CHECKPOINT（預設 v2_best）續訓而非從官方權重開始，因為要測的是
# 「能不能把讀標音這件事加上去」，不是「能不能從頭學台語」。
#
# 預期的副作用是漢字那一側會退步——4.62 小時單一語者的災難性遺忘幾乎必然。
# 所以驗證要同時聽兩側：台羅有沒有學會，漢字退了多少。台羅學會而漢字退步，
# 結論仍然是成立的，後面用 TAT-TTS 的 40.6 小時混合既有語料重做就好。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --job-name=cosyvoice3-suisiann-llm
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"
: "${TP1_INIT_CHECKPOINT:?Set TP1_INIT_CHECKPOINT 指向要續訓的 llm .pt}"
: "${TP1_CONF_SOURCE:?Set TP1_CONF_SOURCE 指向有 taigi_cosyvoice3_v1.yaml 的 conf 目錄}"

tag="${TP1_TAG:-_v5_suisiann}"
tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/llm_ddp${tag}" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/llm_ddp${tag}" \
  "${TP1_WORKSPACE}/conf" \
  "${tp1_cache}"

cp -n "${TP1_CONF_SOURCE}/taigi_cosyvoice3_v1.yaml" "${TP1_WORKSPACE}/conf/" || true

tp1_mounts="${TP1_WORKSPACE}:/workspace"
tp1_mounts+=",${TP1_MODELS}:/models:ro"
tp1_mounts+=",${tp1_cache}:/tmp/cosyvoice-cache"
tp1_mounts+=",${TP1_INIT_CHECKPOINT}:/init/llm.pt:ro"

epochs="${TP1_MAX_EPOCH:-8}"
warmup="${TP1_WARMUP:-200}"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c "
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /init/llm.pt
    test -s /workspace/prepared/data/train/parquet/data.list

    # 原本的設定是給 598 小時用的：warmup 2500 步、跑 20 epoch。這批小兩個
    # 數量級，一個 epoch 只有幾百步，照原設定整段訓練都還在暖身，學習率永遠
    # 到不了設定值。改設定而不是改程式，這樣原本的實驗不受影響。
    python - <<'PY'
import re
from pathlib import Path

path = Path('/workspace/conf/taigi_suisiann.yaml')
source = Path('/workspace/conf/taigi_cosyvoice3_v1.yaml').read_text()
source = re.sub(r'(warmup_steps:\s*)\d+', r'\g<1>${warmup}', source)
source = re.sub(r'(max_epoch:\s*)\d+', r'\g<1>${epochs}', source, count=1)
path.write_text(source)
for line in source.splitlines():
    if re.search(r'warmup_steps|max_epoch|^\s+lr:', line):
        print(line.strip(), flush=True)
PY

    python -c 'import torch; assert torch.cuda.device_count() == 2, torch.cuda.device_count()'

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config /workspace/conf/taigi_suisiann.yaml \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /init/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp${tag} \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp${tag} \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  "
