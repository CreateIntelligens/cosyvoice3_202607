#!/usr/bin/env bash
# Stage B flow fine-tuning. Submit with the partition/account assigned to your
# Taipei-1 project, for example:
# sbatch -p PARTITION --export=ALL,TP1_IMAGE=...,TP1_WORKSPACE=...,TP1_MODELS=... \
#   examples/taigi/cosyvoice3/jobs/taipei1_train_flow.sbatch
#
# One Slurm task owns the node. torchrun starts one worker for each of the 8 GPUs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-flow
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE to an absolute .sqsh path or a Pyxis registry URI}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE to the writable experiment root on Taipei-1}"
: "${TP1_MODELS:?Set TP1_MODELS to the persistent base-model root on Taipei-1}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/flow" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/flow" \
  "${TP1_WORKSPACE}/run-metadata" \
  "${tp1_cache}"

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${tp1_cache}:/tmp/cosyvoice-cache"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /models/Fun-CosyVoice3-0.5B/flow.pt
    test -s /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/config.json
    test -s /workspace/conf/taigi_cosyvoice3_v1.yaml
    test -s /workspace/prepared/data/train/parquet/data.list
    test -s /workspace/prepared/data/dev/parquet/data.list
    test -w /workspace/exp
    python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == 8, torch.cuda.device_count(); print([torch.cuda.get_device_name(i) for i in range(8)])"
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model flow \
      --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /models/Fun-CosyVoice3-0.5B/flow.pt \
      --model_dir /workspace/exp/taigi_v1/flow \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/flow \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  '
