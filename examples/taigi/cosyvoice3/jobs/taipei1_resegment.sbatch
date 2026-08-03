#!/usr/bin/env bash
# Re-cut the drama episodes from the vocal-isolated masters using the broadcast
# cue lists.  Pure CPU: the VAD is an RMS envelope, so this job asks for no GPU
# and can schedule alongside training.
#
#   sbatch -p p06 --export=ALL examples/taigi/cosyvoice3/jobs/taipei1_resegment.sbatch
#
# Set TP1_DRAMA=drama1|drama2 and TP1_MODE=validate|segment.
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=taigi-resegment
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

tp1_image="${TP1_IMAGE:-/mnt/shared/p06/cosyvoice/cosyvoice3-taipei1.sqsh}"
tp1_root="${TP1_ROOT:-/mnt/shared/p06/taigi_resegment}"
tp1_audio="${TP1_AUDIO:-/mnt/shared/p06/uvr5_separated}"
drama="${TP1_DRAMA:-drama1}"
mode="${TP1_MODE:-validate}"
workers="${TP1_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
episodes="${TP1_EPISODES:-}"
stamp="$(date +%Y%m%d-%H%M%S)"

test -s "${tp1_image}"
test -d "${tp1_audio}/${drama}"
test -d "${tp1_root}/cuelists/${drama}"
mkdir -p "${tp1_root}/segments" "${tp1_root}/validation" "${tp1_root}/logs"

srun \
  --container-image "${tp1_image}" \
  --container-mounts "${tp1_root}:/resegment,${tp1_audio}:/audio:ro" \
  bash -c "
    set -euo pipefail
    python -c 'import numpy, soundfile; print(numpy.__version__, soundfile.__version__)'
    args=(
      --audio-dir /audio/${drama}
      --cuelist-dir /resegment/cuelists/${drama}
      --workers ${workers}
    )
    [ -n '${episodes}' ] && args+=(--episodes '${episodes}')
    if [ '${mode}' = validate ]; then
      args+=(
        --reference-dir /resegment/cuelists/corrected_srt/${drama}
        --out-dir /resegment/segments/_validate_${drama}
        --dry-run
        --report /resegment/validation/${drama}-${stamp}.json
      )
    else
      args+=(
        --out-dir /resegment/segments/${drama}
        --report /resegment/logs/segment-${drama}-${stamp}.json
      )
    fi
    python /resegment/tools/make_segments_from_cuelist.py \"\${args[@]}\"
  "
