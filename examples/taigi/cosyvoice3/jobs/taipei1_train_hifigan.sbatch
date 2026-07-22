#!/usr/bin/env bash
# Conservative one-GPU HiFT candidate run. Submit with Taipei-1 allocation values:
# sbatch -p PARTITION -A ACCOUNT --export=ALL,TP1_IMAGE=...,TP1_WORKSPACE=...,TP1_DATASETS=...,TP1_QC=... \
#   examples/taigi/cosyvoice3/jobs/taipei1_train_hifigan.sbatch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-hift
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE to the absolute .sqsh path}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE to the writable experiment root}"
: "${TP1_DATASETS:?Set TP1_DATASETS to the drama1/drama2 parent directory}"
: "${TP1_QC:?Set TP1_QC to the QC JSON directory}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_DATASETS}:/datasets,${TP1_QC}:/qc,${tp1_cache}:/tmp/cosyvoice-cache"
mkdir -p "${TP1_WORKSPACE}/exp/taigi_v1/hifigan" "${TP1_WORKSPACE}/tensorboard/taigi_v1/hifigan" "${tp1_cache}"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice
    test -r /workspace/exp/taigi_v1/hifigan_init.pt
    torchrun --standalone --nnodes=1 --nproc_per_node=1 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model hifigan \
      --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
      --checkpoint /workspace/exp/taigi_v1/hifigan_init.pt \
      --model_dir /workspace/exp/taigi_v1/hifigan \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/hifigan \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory
  '
