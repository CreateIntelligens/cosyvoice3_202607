#!/usr/bin/env bash
set -euo pipefail

# Run inside the CosyVoice image.  The same stages work for the 10-speaker
# local smoke set and the full drama1+drama2 Taipei-1 preparation.
stage="${STAGE:-0}"
stop_stage="${STOP_STAGE:-3}"
mode="${MODE:-smoke}"
cosyvoice_root="${COSYVOICE_ROOT:-/opt/CosyVoice}"
workspace="${WORKSPACE:-/workspace}"
prepared_dir="${PREPARED_DIR:-${workspace}/prepared}"
model_dir="${MODEL_DIR:-/models/Fun-CosyVoice3-0.5B}"
dataset_root="${DATASET_ROOT:-/datasets}"
qc_root="${QC_ROOT:-/qc}"
num_threads="${NUM_THREADS:-4}"

if [[ "${mode}" != "smoke" && "${mode}" != "full" ]]; then
  echo "MODE must be smoke or full" >&2
  exit 2
fi

# Stage 1-3 需要處理 Stage 0 實際產生的每個 split，含選配的 test_unseen_speaker。
list_splits() {
  find "${prepared_dir}/data" -mindepth 1 -maxdepth 1 -type d -exec test -f '{}/wav.scp' \; -print \
    | sed 's|.*/||' | sort
}

if (( stage <= 0 && stop_stage >= 0 )); then
  common_args=(
    --source "drama1=${dataset_root}/drama1"
    --source "drama2=${dataset_root}/drama2"
    --container-source "drama1=${dataset_root}/drama1"
    --container-source "drama2=${dataset_root}/drama2"
    --output-dir "${prepared_dir}"
    --container-output-root "${prepared_dir}"
    --seed 1986
  )
  # NAER/TAT-MOE is optional and shaped differently: speakers hold WAVs directly and
  # the transcripts live in metadata.jsonl rather than beside each clip.
  if [[ -n "${NAER_ROOT:-}" ]]; then
    common_args+=(
      --source "naer=${NAER_ROOT}"
      --container-source "naer=${NAER_ROOT}"
      --transcript-jsonl "naer=${NAER_JSONL:?NAER_ROOT set without NAER_JSONL}:${NAER_JSONL_ROOT:?NAER_ROOT set without NAER_JSONL_ROOT}"
    )
  fi
  if [[ -f "${qc_root}/drama1_audio_qc_full.json" && -f "${qc_root}/drama2_audio_qc_full.json" ]]; then
    common_args+=(
      --qc-report "drama1=${qc_root}/drama1_audio_qc_full.json"
      --qc-report "drama2=${qc_root}/drama2_audio_qc_full.json"
    )
  elif [[ "${REQUIRE_QC_REPORTS:-1}" == "1" ]]; then
    echo "QC reports are required but were not found below ${qc_root}" >&2
    exit 2
  fi

  if [[ "${mode}" == "smoke" ]]; then
    python "${cosyvoice_root}/tools/prepare_taigi_data.py" \
      "${common_args[@]}" \
      --speaker-list drama1=002,003,004,005,006 \
      --speaker-list drama2=001,002,003,004,005 \
      --materialize copy \
      --copy-workers "${COPY_WORKERS:-4}" \
      --split-unit utterance \
      --min-duration 1.5 --max-duration 12.0 \
      --train-ratio 0.75 --dev-ratio 0.125 --test-ratio 0.125 \
      --max-per-speaker 40
  else
    full_materialize="${MATERIALIZE:-copy}"
    # HOLDOUT_SPEAKERS="drama1=101,102 drama2=055" 會把這些說話者整批排除在
    # train/dev/test 之外，另存成 test_unseen_speaker 供 zero-shot 評估。
    holdout_args=()
    for entry in ${HOLDOUT_SPEAKERS:-}; do
      holdout_args+=(--holdout-speakers "${entry}")
    done
    python "${cosyvoice_root}/tools/prepare_taigi_data.py" \
      "${common_args[@]}" \
      "${holdout_args[@]+"${holdout_args[@]}"}" \
      --materialize "${full_materialize}" \
      --copy-workers "${COPY_WORKERS:-16}" \
      --split-unit group \
      --min-duration "${MIN_DURATION:-1.0}" --max-duration "${MAX_DURATION:-30.0}" \
      --train-ratio "${TRAIN_RATIO:-0.98}" \
      --dev-ratio "${DEV_RATIO:-0.01}" \
      --test-ratio "${TEST_RATIO:-0.01}"
  fi
  python "${cosyvoice_root}/tools/validate_cosyvoice_data.py" \
    --prepared-dir "${prepared_dir}" --check-audio
fi

if (( stage <= 1 && stop_stage >= 1 )); then
  test -f "${model_dir}/campplus.onnx"
  for split in $(list_splits); do
    python "${cosyvoice_root}/tools/extract_embedding.py" \
      --dir "${prepared_dir}/data/${split}" \
      --onnx_path "${model_dir}/campplus.onnx" \
      --num_thread "${num_threads}"
  done
fi

if (( stage <= 2 && stop_stage >= 2 )); then
  test -f "${model_dir}/speech_tokenizer_v3.onnx"
  for split in $(list_splits); do
    python "${cosyvoice_root}/tools/extract_speech_token.py" \
      --dir "${prepared_dir}/data/${split}" \
      --onnx_path "${model_dir}/speech_tokenizer_v3.onnx" \
      --num_thread "${num_threads}"
  done
fi

if (( stage <= 3 && stop_stage >= 3 )); then
  if [[ "${mode}" == "smoke" ]]; then
    utts_per_parquet="${UTTS_PER_PARQUET:-128}"
    parquet_processes="${PARQUET_PROCESSES:-1}"
  else
    utts_per_parquet="${UTTS_PER_PARQUET:-1000}"
    parquet_processes="${PARQUET_PROCESSES:-16}"
  fi
  for split in $(list_splits); do
    python "${cosyvoice_root}/tools/make_parquet_list.py" \
      --num_utts_per_parquet "${utts_per_parquet}" \
      --num_processes "${parquet_processes}" \
      --src_dir "${prepared_dir}/data/${split}" \
      --des_dir "${prepared_dir}/data/${split}/parquet"
  done
  python "${cosyvoice_root}/tools/validate_cosyvoice_data.py" \
    --prepared-dir "${prepared_dir}" --check-audio --require-parquet
fi
