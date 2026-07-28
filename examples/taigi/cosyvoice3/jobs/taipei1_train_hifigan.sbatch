#!/usr/bin/env bash
# Conservative HiFT candidate run. Submit with Taipei-1 allocation values:
# sbatch -p PARTITION --export=ALL,TP1_IMAGE=...,TP1_WORKSPACE=...,TP1_MODELS=... \
#   examples/taigi/cosyvoice3/jobs/taipei1_train_hifigan.sbatch
#
# Defaults to one GPU because section 12.4 wants GAN fine-tuning kept conservative.
# To try more, override the allocation and give the run its own output directory so
# it cannot clobber a run already in flight:
#   sbatch --gres=gpu:h100:7 --cpus-per-task=112 --export=ALL,TP1_TAG=_7gpu,... ...
# The worker count follows the allocation, so nothing else needs changing.
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
: "${TP1_MODELS:?Set TP1_MODELS to the persistent base-model root on Taipei-1}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
tp1_tag="${TP1_TAG:-}"
tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${tp1_cache}:/tmp/cosyvoice-cache"
mkdir -p "${TP1_WORKSPACE}/exp/taigi_v1/hifigan${tp1_tag}" \
         "${TP1_WORKSPACE}/tensorboard/taigi_v1/hifigan${tp1_tag}" "${tp1_cache}"

# srun carries the environment into the container, so the inner script can read it.
export TP1_TAG="${tp1_tag}"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/config.json
    test -s "${TP1_GAN_CONFIG:-/workspace/conf/taigi_cosyvoice3_v1_gan.yaml}"
    test -s /workspace/prepared/data/train/parquet/data.list
    test -s /workspace/prepared/data/dev/parquet/data.list
    # The official hift.pt is a bare generator state dict; GAN training expects a
    # generator.*/discriminator.* wrapper. Build it once, never in place.
    if [[ ! -s /workspace/exp/taigi_v1/hifigan_init.pt ]]; then
      python tools/convert_hift_checkpoint.py wrap \
        /models/Fun-CosyVoice3-0.5B/hift.pt \
        /workspace/exp/taigi_v1/hifigan_init.pt
    fi
    test -s /workspace/exp/taigi_v1/hifigan_init.pt
    test -w /workspace/exp
    # One worker per visible GPU, whatever the allocation handed us.
    gpus=$(python -c "import torch; print(torch.cuda.device_count())")
    test "${gpus}" -ge 1
    python -c "import torch; print(f\"{torch.cuda.device_count()} GPU: {torch.cuda.get_device_name(0)}\")"
    torchrun --standalone --nnodes=1 --nproc_per_node="${gpus}" \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model hifigan \
      --config "${TP1_GAN_CONFIG:-/workspace/conf/taigi_cosyvoice3_v1_gan.yaml}" \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /workspace/exp/taigi_v1/hifigan_init.pt \
      --model_dir "/workspace/exp/taigi_v1/hifigan${TP1_TAG:-}" \
      --tensorboard_dir "/workspace/tensorboard/taigi_v1/hifigan${TP1_TAG:-}" \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory
  '
