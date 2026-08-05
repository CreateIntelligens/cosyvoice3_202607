#!/usr/bin/env bash
# 準備 SuíSiann，用來測「模型能不能學會讀台羅標音」。
#
# 這批資料只有 4.62 小時、單一語者，訓不出成品，也不打算訓成品。它要回答的是
# 一個是非題：給模型台羅數字調，它會不會照著念。會的話，未見字的問題就從
# 「補訓練資料」變成「查辭典」——台語連調音節只有 2,269 個，是封閉集合，
# 而漢字不是。不會的話，這條路直接斷，省下後面所有力氣。
#
# 每段音訊出現兩次，一次配漢字、一次配台羅，指向同一個檔案。配對本身就是要
# 教的東西。metadata.jsonl 由 tools/prepare_suisiann.py 產生。
#
# 不走 prepare_data.sh：那支腳本綁死 drama1/drama2 兩個來源和 QC 報告，
# 這裡只有一個來源也沒有 QC，直接呼叫底層工具比改腳本乾淨。
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=cosyvoice3-suisiann-prep
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"
: "${TP1_SUISIANN:?Set TP1_SUISIANN 指向 suisiann_prepared}"
: "${TP1_SUISIANN_AUDIO:?Set TP1_SUISIANN_AUDIO 指向 suisiann（連結的目標）}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
mkdir -p "${TP1_WORKSPACE}/prepared" "${tp1_cache}"

# 音檔目錄要一起掛進去。suisiann_prepared 裡是相對符號連結，指到隔壁的
# suisiann/，容器裡少掛一個，每一條連結都會斷。
tp1_mounts="${TP1_WORKSPACE}:/workspace"
tp1_mounts+=",${TP1_MODELS}:/models:ro"
tp1_mounts+=",${tp1_cache}:/tmp/cosyvoice-cache"
tp1_mounts+=",${TP1_SUISIANN}:/datasets/suisiann_prepared:ro"
tp1_mounts+=",${TP1_SUISIANN_AUDIO}:/datasets/suisiann:ro"

# 映像檔停在 7 月的版本，裡面的 prepare_taigi_data.py 還沒有 --transcript-jsonl，
# 而這批資料的文字全靠它。單檔 bind mount 覆蓋掉容器內的版本，跟
# taipei1_prepare_data.sbatch 用的是同一份 patch，兩邊不會分岔。
tp1_patch_dir="${TP1_PATCH_DIR:-/mnt/shared/p06/cosyvoice/patches}"
for patch in prepare_taigi_data.py validate_cosyvoice_data.py; do
  [[ -s "${tp1_patch_dir}/${patch}" ]] &&
    tp1_mounts+=",${tp1_patch_dir}/${patch}:/opt/CosyVoice/tools/${patch}:ro"
done

srun --container-image "${TP1_IMAGE}" --container-mounts "${tp1_mounts}" bash -c '
  set -euo pipefail
  cd /opt/CosyVoice

  test -s /datasets/suisiann_prepared/metadata.jsonl
  # 連結解不開的話，後面每個階段都會失敗，而且錯誤訊息離原因很遠。
  #
  # 用 -print -quit 而不是管到 head：head 讀夠就關管線，find 收到 SIGPIPE，
  # 在 set -o pipefail 之下整個腳本就以 141 結束——而且什麼訊息都不會留下。
  probe=$(find /datasets/suisiann_prepared/tailo -name "*.wav" -print -quit)
  test -r "${probe}" || { echo "符號連結斷了：${probe}" >&2; exit 2; }
  echo "連結可讀：${probe}"

  python tools/prepare_taigi_data.py \
    --source "suisiann=/datasets/suisiann_prepared" \
    --container-source "suisiann=/datasets/suisiann_prepared" \
    --transcript-jsonl "suisiann=/datasets/suisiann_prepared/metadata.jsonl:/datasets/suisiann_prepared" \
    --output-dir /workspace/prepared \
    --container-output-root /workspace/prepared \
    --materialize none \
    --split-unit utterance \
    --min-duration 1.0 --max-duration 30.0 \
    --train-ratio 0.96 --dev-ratio 0.02 --test-ratio 0.02 \
    --seed 1986

  python tools/validate_cosyvoice_data.py --prepared-dir /workspace/prepared --check-audio

  splits=$(find /workspace/prepared/data -mindepth 1 -maxdepth 1 -type d \
             -exec test -f "{}/wav.scp" \; -print | sed "s|.*/||" | sort)
  for split in ${splits}; do
    python tools/extract_embedding.py --dir "/workspace/prepared/data/${split}" \
      --onnx_path /models/Fun-CosyVoice3-0.5B/campplus.onnx --num_thread 8
    python tools/extract_speech_token.py --dir "/workspace/prepared/data/${split}" \
      --onnx_path /models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx --num_thread 8
    # 這批小，每個 parquet 少裝一點，dev/test 才不會擠成單一檔案。
    python tools/make_parquet_list.py --num_utts_per_parquet 200 --num_processes 8 \
      --src_dir "/workspace/prepared/data/${split}" \
      --des_dir "/workspace/prepared/data/${split}/parquet"
  done

  python tools/validate_cosyvoice_data.py --prepared-dir /workspace/prepared \
    --check-audio --require-parquet

  # 兩種寫法都要進得了訓練集，否則對照就不成立。
  python - <<'"'"'PY'"'"'
import pyarrow.parquet as pq
from pathlib import Path
for split in ("train", "dev"):
    listing = Path(f"/workspace/prepared/data/{split}/parquet/data.list")
    counts = {"hanji": 0, "tailo": 0}
    for line in listing.read_text().split():
        for utt in pq.read_table(line, columns=["utt"])["utt"].to_pylist():
            for key in counts:
                if key in utt:
                    counts[key] += 1
    print(f"{split}: 漢字 {counts['hanji']:,}  台羅 {counts['tailo']:,}", flush=True)
PY
'
