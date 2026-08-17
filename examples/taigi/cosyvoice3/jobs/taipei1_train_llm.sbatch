#!/usr/bin/env bash
# Submit with the partition/account assigned to your Taipei-1 project, for example:
# sbatch -p PARTITION -A ACCOUNT --export=ALL,TP1_IMAGE=...,TP1_WORKSPACE=...,TP1_MODELS=... \
#   examples/taigi/cosyvoice3/jobs/taipei1_train_llm.sbatch
#
# One Slurm task owns the node. torchrun starts one worker per allocated GPU.
#
# The directives below ask for the whole node, and sbatch --gres/--cpus-per-task
# on the command line override them. Asking for all eight means waiting for the
# node to empty completely; p06 has exactly one node shared with other accounts,
# so a smaller request often finishes sooner in wall-clock terms even though the
# training itself takes longer:
#
#   sbatch --gres=gpu:h100:4 --cpus-per-task=64 ... taipei1_train_llm.sbatch
#
# The worker count is read from the allocation rather than hard-coded, so the
# override is the only thing that needs changing.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=128
#SBATCH --time=24:00:00
#SBATCH --job-name=cosyvoice3-taigi-llm
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE to an absolute .sqsh path or a Pyxis registry URI}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE to the writable experiment root on Taipei-1}"
: "${TP1_MODELS:?Set TP1_MODELS to the persistent base-model root on Taipei-1}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p \
  "${TP1_WORKSPACE}/exp/taigi_v1/llm_ddp" \
  "${TP1_WORKSPACE}/tensorboard/taigi_v1/llm_ddp" \
  "${TP1_WORKSPACE}/run-metadata" \
  "${tp1_cache}"

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${tp1_cache}:/tmp/cosyvoice-cache"

# SLURM_GPUS_ON_NODE 反映實際配到的張數，命令列用 --gres 覆寫時會跟著變。
gpus="${TP1_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
echo "配到 ${gpus} 張 GPU"

srun --mpi=pmix \
  --container-image "${TP1_IMAGE}" \
  --container-mounts "${tp1_mounts}" \
  bash -c '
    set -euo pipefail
    cd /opt/CosyVoice
    test -s /models/Fun-CosyVoice3-0.5B/llm.pt
    test -s /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN/config.json
    test -s /workspace/conf/taigi_cosyvoice3_v1.yaml
    test -s /workspace/prepared/data/train/parquet/data.list
    test -s /workspace/prepared/data/dev/parquet/data.list
    test -w /workspace/exp
    python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.device_count() == '"${gpus}"', torch.cuda.device_count(); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
    torchrun --standalone --nnodes=1 --nproc_per_node='"${gpus}"' \
      cosyvoice/bin/train.py \
      --train_engine torch_ddp \
      --model llm \
      --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
      --train_data /workspace/prepared/data/train/parquet/data.list \
      --cv_data /workspace/prepared/data/dev/parquet/data.list \
      --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
      --onnx_path /models/Fun-CosyVoice3-0.5B \
      --checkpoint /models/Fun-CosyVoice3-0.5B/llm.pt \
      --model_dir /workspace/exp/taigi_v1/llm_ddp \
      --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp \
      --ddp.dist_backend nccl \
      --num_workers 8 --prefetch 20 --pin_memory --use_amp
  '
