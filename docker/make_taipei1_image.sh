#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ ! -f third_party/Matcha-TTS/matcha/__init__.py ]]; then
  echo "ERROR: Matcha-TTS submodule is missing." >&2
  echo "Run: git submodule update --init --recursive" >&2
  exit 2
fi

source_commit="$(git rev-parse HEAD)"
short_commit="${source_commit:0:12}"
if [[ -n "$(git status --porcelain --untracked-files=normal)" ]]; then
  source_dirty=true
  default_tag="${short_commit}-dirty"
else
  source_dirty=false
  default_tag="${short_commit}"
fi
image_ref="${1:-cosyvoice3-taipei1:${default_tag}}"

echo "Building ${image_ref} from commit ${source_commit}"
docker build \
  --file docker/Dockerfile.taipei1 \
  --build-arg "SOURCE_COMMIT=${source_commit}" \
  --build-arg "SOURCE_DIRTY=${source_dirty}" \
  --tag "${image_ref}" \
  .

echo "Built ${image_ref}"
docker image inspect "${image_ref}" \
  --format 'image_id={{.Id}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
