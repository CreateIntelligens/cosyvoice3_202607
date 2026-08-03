#!/usr/bin/env bash
# 匯出 v3 的四個實驗：每個都取 CV loss 最低的單一 epoch 與最佳五個 epoch 的平均。
# 純 CPU，可以和 GPU 訓練並行。
#
#   sbatch -p p06 --export=ALL examples/taigi/cosyvoice3/jobs/taipei1_export_checkpoints_v3.sbatch
#
# 原本的 taipei1_export_checkpoints.sbatch 把目錄名寫死成 llm_ddp / flow，
# 而多實驗的目錄帶 TP1_TAG 後綴，所以另外寫這一支。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=cosyvoice3-taigi-export-v3
#SBATCH --output=/mnt/shared/p06/cosyvoice/logs/%x-%j.out
#SBATCH --error=/mnt/shared/p06/cosyvoice/logs/%x-%j.err

set -euo pipefail

: "${TP1_IMAGE:?}"
: "${TP1_WORKSPACE:?}"

mkdir -p "${TP1_WORKSPACE}/export"

tp1_mounts="${TP1_WORKSPACE}:/workspace"
patch=/mnt/shared/p06/cosyvoice/patches/average_model.py
[[ -s "${patch}" ]] && tp1_mounts="${tp1_mounts},${patch}:/opt/CosyVoice/cosyvoice/bin/average_model.py:ro"

srun --container-image "${TP1_IMAGE}" --container-mounts "${tp1_mounts}" bash -c '
  set -euo pipefail
  cd /opt/CosyVoice
  # 「來源目錄:輸出前綴」——輸出名稱直接對應實驗代號，免得試聽時搞混
  for pair in \
      llm_ddp_v3a:llm_v3a \
      llm_ddp_v3b_lr2e5:llm_v3b_lr2e5 \
      llm_ddp_v3d_drama:llm_v3d_drama \
      llm_ddp_v3e_naer:llm_v3e_naer \
      llm_ddp_v3g_taskinstr:llm_v3g_taskinstr \
      flow_v3c:flow_v3c ; do
    src=/workspace/exp/taigi_v1/${pair%%:*}
    name=${pair##*:}
    if [[ ! -d ${src} ]]; then echo "跳過 ${name}：${src} 不存在"; continue; fi
    if ! ls ${src}/epoch_*_whole.yaml >/dev/null 2>&1; then
      echo "跳過 ${name}：沒有 epoch checkpoint"; continue
    fi
    for num in 1 5; do
      tag=$([[ ${num} == 1 ]] && echo best || echo avg5)
      echo "=== ${name}_${tag} ==="
      python cosyvoice/bin/average_model.py \
        --dst_model /workspace/export/${name}_${tag}.pt \
        --src_path ${src} --num ${num} --val_best
    done
  done
  echo; echo "=== 產出 ==="; ls -la /workspace/export/
'
