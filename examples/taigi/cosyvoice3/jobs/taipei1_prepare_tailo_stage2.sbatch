#!/usr/bin/env bash
# 第二階段的資料準備：把一份或多份「漢字／台羅雙寫法」語料做成 parquet。
#
# v5/v7 用媠聲（4.62 h）證明了模型學得會讀台羅，也證明混排是零樣本成立的。
# 這一支是把同樣的作法放大：
#
#   v8 = 媠聲 4.62 h ＋ 教育部辭典 28.4 h   兩位朗讀者，33 h
#   v9 = 媠聲 4.62 h ＋ naer 台羅 192.6 h   384 位語者，197 h
#
# v8 的教育部資料是 CC BY-NC-ND（非商業、禁止改作），**訓練出來的模型不得公開
# 發布或商業使用**，除非教育部書面同意。v9 完全沒有這個限制。兩者若打平，就用
# v9，不必等回信——這個對照本身比 v8 更有價值。
#
# 多份語料以空白分隔的 TP1_STAGE2_SOURCES 指定，格式「名稱:主機路徑」：
#
#   TP1_STAGE2_SOURCES="suisiann:/mnt/.../suisiann_prepared moe:/mnt/.../moe_prepared"
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --job-name=cosyvoice3-tailo-prep
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_WORKSPACE:?Set TP1_WORKSPACE}"
: "${TP1_MODELS:?Set TP1_MODELS}"
: "${TP1_STAGE2_SOURCES:?Set TP1_STAGE2_SOURCES=\"名稱:路徑 名稱:路徑\"}"

tp1_cache="${TP1_CACHE:-${TP1_WORKSPACE}/cache}"
# prepared/ 不預先建立：prepare_taigi_data.py 拒絕寫進已存在的目錄，先 mkdir
# 只會讓 job 拿到節點之後才失敗。
mkdir -p "${TP1_WORKSPACE}" "${tp1_cache}"
if [[ -e "${TP1_WORKSPACE}/prepared" ]]; then
  echo "已存在 ${TP1_WORKSPACE}/prepared —— 要重跑請先刪除：" >&2
  echo "  rm -rf ${TP1_WORKSPACE}/prepared" >&2
  exit 2
fi

tp1_mounts="${TP1_WORKSPACE}:/workspace,${TP1_MODELS}:/models:ro,${tp1_cache}:/tmp/cosyvoice-cache"

# 每份語料掛成 /datasets/<名稱>，並組出 prepare_taigi_data.py 的參數。
prep_args=()
for entry in ${TP1_STAGE2_SOURCES}; do
  name="${entry%%:*}"; path="${entry#*:}"
  [[ -d "${path}" ]] || { echo "找不到 ${name} 的目錄：${path}" >&2; exit 2; }
  tp1_mounts+=",${path}:/datasets/${name}:ro"
  prep_args+=(--source "${name}=/datasets/${name}")
  prep_args+=(--container-source "${name}=/datasets/${name}")
  prep_args+=(--transcript-jsonl "${name}=/datasets/${name}/metadata.jsonl:/datasets/${name}")
done
export TP1_PREP_ARGS="${prep_args[*]}"

# 映像檔停在 7 月，裡面的 prepare_taigi_data.py 還沒有 --transcript-jsonl。
tp1_patch_dir="${TP1_PATCH_DIR:-/mnt/shared/p06/cosyvoice/patches}"
for patch in prepare_taigi_data.py validate_cosyvoice_data.py; do
  [[ -s "${tp1_patch_dir}/${patch}" ]] &&
    tp1_mounts+=",${tp1_patch_dir}/${patch}:/opt/CosyVoice/tools/${patch}:ro"
done

srun --container-image "${TP1_IMAGE}" --container-mounts "${tp1_mounts}" bash -c '
  set -euo pipefail
  cd /opt/CosyVoice

  for entry in ${TP1_STAGE2_SOURCES}; do
    name="${entry%%:*}"
    test -s "/datasets/${name}/metadata.jsonl"
    # 連結解不開的話後面每個階段都會失敗，而錯誤訊息離原因很遠。
    # -print -quit 而不是管到 head：head 讀夠就關管線，find 收到 SIGPIPE，
    # 在 set -o pipefail 之下整個腳本會以 141 結束而且什麼都不留。
    probe=$(find "/datasets/${name}" -name "*.wav" -print -quit)
    test -r "${probe}" || { echo "讀不到 ${name} 的音檔：${probe}" >&2; exit 2; }
    echo "${name}: $(wc -l < /datasets/${name}/metadata.jsonl) 筆"
  done

  python tools/prepare_taigi_data.py ${TP1_PREP_ARGS} \
    --output-dir /workspace/prepared \
    --container-output-root /workspace/prepared \
    --materialize none \
    --split-unit utterance \
    --min-duration 1.0 --max-duration 30.0 \
    --train-ratio 0.98 --dev-ratio 0.01 --test-ratio 0.01 \
    --seed 1986

  python tools/validate_cosyvoice_data.py --prepared-dir /workspace/prepared --check-audio

  splits=$(find /workspace/prepared/data -mindepth 1 -maxdepth 1 -type d \
             -exec test -f "{}/wav.scp" \; -print | sed "s|.*/||" | sort)
  for split in ${splits}; do
    python tools/extract_embedding.py --dir "/workspace/prepared/data/${split}" \
      --onnx_path /models/Fun-CosyVoice3-0.5B/campplus.onnx --num_thread 8
    python tools/extract_speech_token.py --dir "/workspace/prepared/data/${split}" \
      --onnx_path /models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx --num_thread 8
    python tools/make_parquet_list.py --num_utts_per_parquet 1000 --num_processes 16 \
      --src_dir "/workspace/prepared/data/${split}" \
      --des_dir "/workspace/prepared/data/${split}/parquet"
  done

  python tools/validate_cosyvoice_data.py --prepared-dir /workspace/prepared \
    --check-audio --require-parquet

  # 兩種寫法都要進得了訓練集，否則對照不成立。這裡不用單引號：整段包在
  # bash -c '"'"'...'"'"' 裡，一個裸單引號就會提早結束外層字串。
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
    han, tai = counts["hanji"], counts["tailo"]
    print(f"{split}: 漢字 {han:,}  台羅 {tai:,}", flush=True)
PY
'
