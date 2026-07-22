#!/usr/bin/env bash
# Submit with the partition/account assigned to your Taipei-1 project, for example:
# sbatch -p PARTITION -A ACCOUNT --export=ALL,TP1_IMAGE=...,TP1_WORKSPACE=...,TP1_DATASETS=...,TP1_QC=... \
#   examples/taigi/cosyvoice3/jobs/taipei1_train_llm.sbatch
#
# One Slurm task owns the node. torchrun starts one worker for each of the 8 GPUs.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-llm
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE to an absolute .sqsh path or a Pyxis registry URI}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE to the writable experiment root on Taipei-1}"
: "${TP1_DATASETS:?Set TP1_DATASETS to the drama1/drama2 parent directory}"
: "${TP1_QC:?Set TP1_QC to the QC JSON directory}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/llm_ddp" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/llm_ddp" \
  "${TP1_WORKSPACE}/run-metadata" \
  "${tp1_cache}"

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_DATASETS}:/datasets,${TP1_QC}:/qc,${tp1_cache}:/tmp/cosyvoice-cache"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice
    test -r /workspace/models/Fun-CosyVoice3-0.5B/llm.pt
    test -r /workspace/prepared/data/train/parquet/data.list
    test -r /workspace/prepared/data/dev/parquet/data.list
    test -w /workspace/exp
    python tools/container_smoke_test.py --cpu-only
    torchrun --standalone --nnodes=1 --nproc_per_node=8 \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
      --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp \
      --ddp.dist_backend nccl \
      --num_workers 4 --prefetch 20 --pin_memory --use_amp
  '
