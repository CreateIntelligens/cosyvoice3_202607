#!/usr/bin/env bash
# 在叢集上直接合成，只把音檔拉回來。
#
# 一個 checkpoint 是 1.89 GB，而那條 tunnel 只有 5.2 MB/s——光下載就六分鐘。
# 要比較四五個 epoch 的話，在這邊合成再拉幾百 KB 的 wav 快得多。
#
# 用途是找出第二階段從第幾個 epoch 開始把底模弄壞。已知 epoch 3 的輸出詞彙
# 已經錯了（「東西」念成 tang-si 而非物件 mih8-kiann7），所以要看更早的。
#
#   TP1_EXP_DIR=.../llm_ddp_v8b_moe_restricted \
#   TP1_EPOCHS="0 1 2" \
#   TP1_TEXT="明天早上九點，我們一起去市場買東西。" \
#   sbatch -p p06 --gres=gpu:h100:1 ... taipei1_synth_epochs.sbatch
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --job-name=cosyvoice3-synth-epochs
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

: "${TP1_IMAGE:?Set TP1_IMAGE}"
: "${TP1_MODELS:?Set TP1_MODELS}"
: "${TP1_EXP_DIR:?Set TP1_EXP_DIR 指向含 epoch_N_whole.pt 的目錄}"
: "${TP1_PROMPT:?Set TP1_PROMPT 指向參考音檔}"
: "${TP1_OUT:?Set TP1_OUT 音檔輸出目錄}"

epochs="${TP1_EPOCHS:-0 1 2}"
text="${TP1_TEXT:-明天早上九點，我們一起去市場買東西。}"
seeds="${TP1_SEEDS:-1942676902 1320383193}"

mkdir -p "${TP1_OUT}"
tp1_mounts="${TP1_MODELS}:/models:ro"
tp1_mounts+=",${TP1_EXP_DIR}:/ckpt:ro"
tp1_mounts+=",${TP1_PROMPT}:/prompt.wav:ro"
tp1_mounts+=",${TP1_OUT}:/out"

export TP1_EPOCHS="${epochs}" TP1_TEXT="${text}" TP1_SEEDS="${seeds}"

srun --container-image "${TP1_IMAGE}" --container-mounts "${tp1_mounts}" bash -c '
  set -euo pipefail
  cd /opt/CosyVoice
  python - <<'"'"'PY'"'"'
import os, random, shutil, sys
from pathlib import Path
import numpy as np, torch, torchaudio
sys.path.insert(0, "/opt/CosyVoice")
sys.path.insert(0, "/opt/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

BASE = Path("/models/Fun-CosyVoice3-0.5B")
OUT = Path("/out")
# 多段文字用 || 分隔，一次跑完省下重複載入模型的時間
TEXTS = [t for t in os.environ["TP1_TEXT"].split("||") if t.strip()]
EPOCHS = os.environ["TP1_EPOCHS"].split()
SEEDS = [int(s) for s in os.environ["TP1_SEEDS"].split()]
INSTRUCT = ("You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。"
            "<|endofprompt|>")

def variant_dir(ckpt: Path, tag: str) -> Path:
    """CosyVoice3 只認一個目錄，所以把官方模型連結過來、llm.pt 換成要測的那個。"""
    target = Path(f"/tmp/variant_{tag}")
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in BASE.iterdir():
        if entry.name != "llm.pt":
            (target / entry.name).symlink_to(entry)
    # epoch checkpoint 帶著 step/epoch 兩個鍵，模型載入器不吃，先去掉。
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    torch.save({k: v for k, v in state.items() if k not in ("step", "epoch")},
               target / "llm.pt")
    return target

for i, t in enumerate(TEXTS, 1):
    print(f"文字 {i}：{t}", flush=True)
for epoch in EPOCHS:
    ckpt = Path("/ckpt") / f"epoch_{epoch}_whole.pt"
    if not ckpt.is_file():
        print(f"  ep{epoch} 找不到 {ckpt}", flush=True)
        continue
    model = AutoModel(model_dir=str(variant_dir(ckpt, epoch)))
    for index, text in enumerate(TEXTS, 1):
        for seed in SEEDS:
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            chunks = [o["tts_speech"] for o in model.inference_instruct2(
                text, INSTRUCT, "/prompt.wav", stream=False, speed=1.0,
                text_frontend=False)]
            audio = torch.concat(chunks, dim=1)
            path = OUT / f"ep{int(epoch):02d}__文字{index}__seed{seed}.wav"
            torchaudio.save(str(path), audio, model.sample_rate)
            print(f"  ep{epoch} 文字{index} seed {seed}  "
                  f"{audio.shape[1] / model.sample_rate:.2f}s → {path.name}", flush=True)
    del model
    torch.cuda.empty_cache()
PY
'
