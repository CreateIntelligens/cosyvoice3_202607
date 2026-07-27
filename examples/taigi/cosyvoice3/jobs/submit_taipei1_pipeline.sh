#!/usr/bin/env bash
# Submit the whole Taipei-1 chain: data prep (stage 0-3) -> LLM -> flow -> HiFi-GAN.
#
# Each stage only starts when the previous one exits cleanly, so a failure stops
# the chain instead of burning H100 hours on missing Parquet. The partition owns
# a single node, so the stages run one after another rather than in parallel.
#
#   bash submit_taipei1_pipeline.sh
#   SKIP_PREP=1 bash submit_taipei1_pipeline.sh     # reuse existing Parquet
#   STOP_AFTER=llm bash submit_taipei1_pipeline.sh  # prep + LLM only
set -euo pipefail

partition="${TP1_PARTITION:-p06}"
root="${TP1_ROOT:-/mnt/shared/p06/cosyvoice}"
jobs_dir="${TP1_JOBS_DIR:-${root}/jobs}"
logdir="${TP1_LOGDIR:-${root}/logs}"

export TP1_IMAGE="${TP1_IMAGE:-${root}/cosyvoice3-taipei1.sqsh}"
export TP1_WORKSPACE="${TP1_WORKSPACE:-${root}/taigi_v1}"
export TP1_MODELS="${TP1_MODELS:-${root}/pretrained_models}"
export TP1_DATASETS="${TP1_DATASETS:-/mnt/shared/p06/dataset202607_1/tai8/segments}"
export TP1_QC="${TP1_QC:-${root}/qc}"

# Episode-level splitting means the ratios are applied to episodes, not
# utterances. See docs/taigi-universal-tts-training.md section 6.
export TRAIN_RATIO="${TRAIN_RATIO:-0.90}"
export DEV_RATIO="${DEV_RATIO:-0.05}"
export TEST_RATIO="${TEST_RATIO:-0.05}"
export HOLDOUT_SPEAKERS="${HOLDOUT_SPEAKERS:-drama1=680,019,489,038 drama2=010,011}"

mkdir -p "${logdir}"

submit() {
  local name="$1" script="$2" after="${3:-}"
  local args=(-p "${partition}" -o "${logdir}/%x-%j.out" -e "${logdir}/%x-%j.err" --export=ALL)
  [[ -n "${after}" ]] && args+=(--dependency="afterok:${after}")
  local id
  id=$(sbatch "${args[@]}" "${jobs_dir}/${script}" | awk '{print $NF}')
  echo "  ${name}: job ${id}${after:+  (waits for ${after})}" >&2
  echo "${id}"
}

echo "提交鏈：" >&2
dep=""
if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  dep=$(submit "prep (stage 0-3)" taipei1_prepare_data.sbatch)
fi
[[ "${STOP_AFTER:-}" == "prep" ]] && exit 0

dep=$(submit "train llm" taipei1_train_llm.sbatch "${dep}")
[[ "${STOP_AFTER:-}" == "llm" ]] && exit 0

dep=$(submit "train flow" taipei1_train_flow.sbatch "${dep}")
[[ "${STOP_AFTER:-}" == "flow" ]] && exit 0

submit "train hifigan" taipei1_train_hifigan.sbatch "${dep}" >/dev/null

echo >&2
echo "監看： squeue -u \$USER    日誌： ${logdir}/" >&2
