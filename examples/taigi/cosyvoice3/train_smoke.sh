#!/usr/bin/env bash
set -euo pipefail

cosyvoice_root="${COSYVOICE_ROOT:-/opt/CosyVoice}"
workspace="${WORKSPACE:-/workspace}"
prepared_dir="${PREPARED_DIR:-${workspace}/prepared}"
model_dir="${MODEL_DIR:-${workspace}/models/Fun-CosyVoice3-0.5B}"
config="${CONFIG:-${workspace}/conf/cosyvoice3_smoke.yaml}"
train_engine="${TRAIN_ENGINE:-deepspeed}"
deepspeed_config="${DEEPSPEED_CONFIG:-${cosyvoice_root}/examples/taigi/cosyvoice3/conf/ds_stage2_cpu_offload.json}"
model="${MODEL:-llm}"
num_gpus="${NUM_GPUS:-1}"
max_epoch="${MAX_EPOCH:-1}"
max_frames="${MAX_FRAMES_IN_BATCH:-100}"

if [[ "${model}" != "llm" && "${model}" != "flow" ]]; then
  echo "MODEL must be llm or flow; HiFi-GAN requires a converted wrapper checkpoint" >&2
  exit 2
fi

if [[ "${train_engine}" != "torch_ddp" && "${train_engine}" != "deepspeed" ]]; then
  echo "TRAIN_ENGINE must be torch_ddp or deepspeed" >&2
  exit 2
fi

mkdir -p "$(dirname "${config}")"
if [[ ! -f "${config}" ]]; then
  cp "${cosyvoice_root}/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml" "${config}"
  sed -i "s/max_frames_in_batch: 2000/max_frames_in_batch: ${max_frames}/" "${config}"
  sed -i "s/max_epoch: 200/max_epoch: ${max_epoch}/g" "${config}"
  sed -i "s/accum_grad: 2/accum_grad: 1/" "${config}"
  sed -i "s/log_interval: 100/log_interval: 1/g" "${config}"
fi

test -f "${prepared_dir}/data/train/parquet/data.list"
test -f "${prepared_dir}/data/dev/parquet/data.list"
test -f "${model_dir}/${model}.pt"

engine_args=()
if [[ "${train_engine}" == "deepspeed" ]]; then
  test -f "${deepspeed_config}"
  engine_args+=(--deepspeed_config "${deepspeed_config}")
  engine_args+=(--deepspeed.save_states "${DEEPSPEED_SAVE_STATES:-model_only}")
else
  engine_args+=(--use_amp)
fi
torchrun --standalone --nnodes=1 --nproc_per_node="${num_gpus}" \
  "${cosyvoice_root}/cosyvoice/bin/train.py" \
  --train_engine "${train_engine}" \
  --model "${model}" \
  --config "${config}" \
  --train_data "${prepared_dir}/data/train/parquet/data.list" \
  --cv_data "${prepared_dir}/data/dev/parquet/data.list" \
  --qwen_pretrain_path "${model_dir}/CosyVoice-BlankEN" \
  --onnx_path "${model_dir}" \
  --checkpoint "${model_dir}/${model}.pt" \
  --model_dir "${workspace}/exp/${model}_${train_engine}" \
  --tensorboard_dir "${workspace}/tensorboard/${model}_${train_engine}" \
  --ddp.dist_backend nccl \
  --num_workers "${NUM_WORKERS:-1}" \
  --prefetch "${PREFETCH:-2}" \
  --pin_memory \
  "${engine_args[@]}"
