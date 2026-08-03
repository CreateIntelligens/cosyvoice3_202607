#!/usr/bin/env bash
# Stage 0-3 資料準備：manifest -> embedding -> speech token -> Parquet
#
#   export HOLDOUT_SPEAKERS="drama1=0037,0198 drama2=1750,2100"   # 值含逗號，必須先 export
#   sbatch -p p06 --export=ALL examples/taigi/cosyvoice3/jobs/taipei1_prepare_data.sbatch
#
# TP1_QC 是選配。QC 報告以相對 WAV 路徑為鍵，換一份切割就完全對不上，
# 這時留空即可（prepare_data.sh 會以 REQUIRE_QC_REPORTS=0 略過）。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-prep
#SBATCH --output=/mnt/shared/p06/cosyvoice/logs/%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE to the absolute .sqsh path}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE to the writable experiment root}"
: "${TP1_MODELS:?Set TP1_MODELS to the persistent base-model root}"
: "${TP1_DATASETS:?Set TP1_DATASETS to the segments root (contains drama1/ drama2/)}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
# prepare_taigi_data.py refuses to write into an existing --output-dir, so
# /workspace/prepared must NOT be created here.
mkdir -p "${TP1_WORKSPACE}/run-metadata" "${tp1_cache}"

# Taipei-1 的 Pyxis 已實測支援 :ro 第三欄（2026-07-27 驗證）
tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${TP1_DATASETS}:/datasets:ro,${tp1_cache}:/tmp/cosyvoice-cache"

require_qc=0
if [[ -n "${TP1_QC:-}" ]]; then
  tp1_mounts="${tp1_mounts},${TP1_QC}:/qc:ro"
  require_qc=1
fi

# NAER/TAT-MOE is optional. naer_flat holds one symlink per speaker so the three
# official partitions look like a single corpus root; the links point at /naer, so
# both mounts have to be present together.
if [[ -n "${TP1_NAER:-}" && -n "${TP1_NAER_FLAT:-}" ]]; then
  tp1_mounts="${tp1_mounts},${TP1_NAER}:/naer:ro,${TP1_NAER_FLAT}:/naer_flat:ro"
  # 換一份轉寫（例如台文改寫成華語）時，只要覆蓋這一個檔案就好：音檔、
  # naer_flat 的連結、utterance ID 全部不變，原始語料一個位元組都不會動到。
  if [[ -n "${TP1_NAER_JSONL:-}" ]]; then
    test -s "${TP1_NAER_JSONL}"
    tp1_mounts="${tp1_mounts},${TP1_NAER_JSONL}:/naer/metadata.jsonl:ro"
  fi
fi

# image 建於 commit 7526c48，缺少 INSTRUCT 修正（文件 §5：訓練與推論 instruction 必須
# 完全一致）與 --holdout-speakers。用修正版覆蓋容器內對應檔案；image 重建後可移除整段。
tp1_patch_dir="${TP1_PATCH_DIR:-/mnt/shared/p06/cosyvoice/patches}"
for patch in prepare_taigi_data.py validate_cosyvoice_data.py; do
  [[ -s "${tp1_patch_dir}/${patch}" ]] &&
    tp1_mounts="${tp1_mounts},${tp1_patch_dir}/${patch}:/opt/CosyVoice/tools/${patch}:ro"
done
[[ -s "${tp1_patch_dir}/prepare_data.sh" ]] &&
  tp1_mounts="${tp1_mounts},${tp1_patch_dir}/prepare_data.sh:/opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh:ro"

export REQUIRE_QC_REPORTS="${require_qc}"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice

    # preflight
    test -r /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
    test -d /datasets/drama1
    test -d /datasets/drama2
    if [[ "${REQUIRE_QC_REPORTS}" == "1" ]]; then
      test -s /qc/drama1_audio_qc_full.json
      test -s /qc/drama2_audio_qc_full.json
    fi
    test -s /models/Fun-CosyVoice3-0.5B/campplus.onnx
    test -s /models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx
    test -w /workspace
    if [[ -n "${TP1_NAER:-}" ]]; then
      test -d /naer_flat && test -s /naer/metadata.jsonl
      echo "naer speakers: $(ls /naer_flat | wc -l)"
    fi
    python -c "import onnxruntime; assert \"CUDAExecutionProvider\" in onnxruntime.get_available_providers(), onnxruntime.get_available_providers()"
    echo "datasets: drama1=$(ls /datasets/drama1 | wc -l) drama2=$(ls /datasets/drama2 | wc -l) groups"
    echo "holdout: ${HOLDOUT_SPEAKERS:-<none>}"

    MODE=full \
    MATERIALIZE="${MATERIALIZE:-copy}" \
    HOLDOUT_SPEAKERS="${HOLDOUT_SPEAKERS:-}" \
    REQUIRE_QC_REPORTS="${REQUIRE_QC_REPORTS}" \
    NAER_ROOT="${TP1_NAER:+/naer_flat}" \
    NAER_JSONL="${TP1_NAER:+/naer/metadata.jsonl}" \
    NAER_JSONL_ROOT="${TP1_NAER:+/naer}" \
    TRAIN_RATIO="${TRAIN_RATIO:-0.90}" \
    DEV_RATIO="${DEV_RATIO:-0.05}" \
    TEST_RATIO="${TEST_RATIO:-0.05}" \
    MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
    DATASET_ROOT=/datasets \
    QC_ROOT=/qc \
    PREPARED_DIR=/workspace/prepared \
    NUM_THREADS="${NUM_THREADS:-16}" \
    COPY_WORKERS="${COPY_WORKERS:-16}" \
    STAGE="${STAGE:-0}" STOP_STAGE="${STOP_STAGE:-3}" \
    bash examples/taigi/cosyvoice3/prepare_data.sh

    # 產出檢查
    test -s /workspace/prepared/data/train/parquet/data.list
    test -s /workspace/prepared/data/dev/parquet/data.list
    wc -l /workspace/prepared/data/*/parquet/data.list
  '
