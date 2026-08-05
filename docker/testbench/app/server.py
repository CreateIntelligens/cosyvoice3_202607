"""Mix-and-match CosyVoice3 checkpoints and listen to the difference.

CosyVoice3 loads llm.pt / flow.pt / hift.pt from a single directory, so a
variant is materialised as a directory of symlinks: everything comes from the
official model except the parts the caller picked. That keeps the stock loader
untouched and costs no disk.

Only one model is resident at a time. The three modules together need several GB
of VRAM and this box shares its GPU with other services, so switching variants
frees the previous one first.
"""
from __future__ import annotations

import io
import json
import logging
import os
import secrets
import shutil
import sys
import threading
from pathlib import Path
from urllib.parse import quote

import soundfile as sf
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/opt/CosyVoice")
sys.path.insert(0, COSYVOICE_ROOT)
sys.path.insert(0, f"{COSYVOICE_ROOT}/third_party/Matcha-TTS")

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("testbench")

MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/models"))
BASE_DIR = MODELS_ROOT / "base"
VARIANTS_DIR = MODELS_ROOT / "variants"
PROMPTS_DIR = MODELS_ROOT / "prompts"
WORK_DIR = Path(os.environ.get("VARIANT_WORK_DIR", "/tmp/cosyvoice-variants"))
STATIC_DIR = Path(__file__).parent / "static"

PARTS = ("llm", "flow", "hift")
OFFICIAL = "official"
# The averaged checkpoints never win a listening test here, and every extra entry
# is one more thing to mis-click during an A/B. The files stay on disk — drop the
# suffix from this tuple to bring them back.
HIDDEN_SUFFIXES = ("_avg5",)

# A bare filename says nothing about what the run was testing, and after a dozen
# experiments that is the difference between a meaningful comparison and a
# guess. Anything not listed here still shows, just without a note.
MODEL_NOTES = {
    # llm
    "llm_best": "v1 · 戲劇 190h",
    "llm_avg5": "v1 · 戲劇 190h · 平均",
    "llm_v2_best": "v2 · tai8+naer 324h · 目前基準",
    "llm_v2_avg5": "v2 · tai8+naer 324h · 平均",
    "llm_v3a_best": "v3-A · 重切戲劇+naer 598h",
    "llm_v3a_avg5": "v3-A · 598h · 平均",
    "llm_v3a_ep5": "v3-A · 598h · epoch 5 原始檔",
    "llm_v3b_lr2e5_best": "v3-B · 同 A 但 lr 2e-5 · 已知較差",
    "llm_v3b_lr2e5_avg5": "v3-B · lr 2e-5 · 平均",
    "llm_v3d_drama_best": "v3-D · 只有戲劇 384h · 半數講中文",
    "llm_v3d_drama_avg5": "v3-D · 只有戲劇 · 平均",
    "llm_v3d_drama_ep5": "v3-D · 只有戲劇 · epoch 5 原始檔",
    "llm_v3e_naer_best": "v3-E · 只有 naer 214h",
    "llm_v3e_naer_avg5": "v3-E · 只有 naer · 平均",
    "llm_v3g_taskinstr_best": "v3-G · 各語料標各自 instruction · 詞內混雜",
    "llm_v3g_taskinstr_avg5": "v3-G · 分別標 instruction · 平均",
    "llm_v4_naerzh_best": "v4 · tai8+naer(文字轉華語) · 對照 v2",
    "llm_v4_naerzh_avg5": "v4 · naer 轉華語 · 平均",
    # v5 是概念驗證，不是候選成品：4.62 小時單一語者，從 v2 續訓，要測的是
    # 給它台羅數字調（khian5-khun1）會不會照著念。漢字那側預期退步。
    "llm_v5_suisiann_best": "v5 · 媠聲 4.6h 教台羅標音 · 驗證用，非成品",
    "llm_v5_suisiann_avg5": "v5 · 媠聲 · 平均 · 驗證用",
    # flow
    "flow_best": "v1 · 戲劇 190h",
    "flow_avg5": "v1 · 戲劇 190h · 平均",
    "flow_v2_best": "v2 · tai8+naer 324h",
    "flow_v2_avg5": "v2 · tai8+naer · 平均",
    "flow_v3c_best": "v3-C · 598h · 與 official 幾無差異",
    "flow_v3c_avg5": "v3-C · 598h · 平均",
    # hift
    "hift_taigi_7gpu_e0": "7 卡實驗 · epoch 0",
    "hift_taigi_7gpu_e9": "7 卡實驗 · epoch 9",
}
# torch accepts a much wider range, but 32-bit keeps the value short enough to
# quote in a bug report and matches what the UI offers.
MAX_SEED = 2 ** 31 - 1
# frontend.py asserts this before extracting speech tokens; catching it here turns
# an opaque 500 into something the caller can act on.
MAX_PROMPT_SEC = 30.0
MIN_PROMPT_SEC = 1.0
# CosyVoice3 tokenises this to id 151646 and asserts on its presence; without it the
# LLM thread dies before emitting a single token.
END_OF_PROMPT = "<|endofprompt|>"
DEFAULT_INSTRUCT = f"You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。{END_OF_PROMPT}"

# Qwen 的詞彙表偏簡體：繁體字沒有共用字形就退化成 UTF-8 位元組，訓練文字裡
# 20.6% 的相異字都這樣（嬤、諒、蹤、廬…），模型從未把它們當成一個字看待。
# 換成簡體字形後多數變回完整 token，實測「三顧茅廬」在三個 seed 都由全錯轉為正確。
#
# 但全域繁轉簡會把 颱/臺/台/檯 併成「台」、「慾」併成台文虛詞「欲」，製造新的歧義。
# 這張表只收「會碎片、簡體完整、且不與語料中其他字合併」的字，由
# tools/build_token_safe_map.py 產生，可直接開啟檢視。
T2S_SAFE_PATH = Path(os.environ.get(
    "T2S_SAFE_MAP", "/opt/CosyVoice/tools/data/t2s_safe.json"))
try:
    T2S_SAFE = str.maketrans(json.loads(T2S_SAFE_PATH.read_text(encoding="utf-8")))
    T2S_COUNT = len(json.loads(T2S_SAFE_PATH.read_text(encoding="utf-8")))
except Exception:
    T2S_SAFE, T2S_COUNT = {}, 0


def _load_map(path: Path) -> tuple[dict, int]:
    try:
        table = json.loads(path.read_text(encoding="utf-8"))
        return str.maketrans(table), len(table)
    except Exception:
        return {}, 0


T2S_FULL, T2S_FULL_COUNT = _load_map(T2S_SAFE_PATH.with_name("t2s_full.json"))

# 逐字轉換會把「乾坤」寫成「干坤」，因為簡化並沒有動 qián 那個讀音。詞組表
# （OpenCC 的 TSPhrases，277 條、其中 61 條是恆等）就是用來保護這類詞的：
# 先做最長比對，命中就照抄，剩下的字才逐字轉。
try:
    T2S_PHRASES: dict[str, str] = json.loads(
        T2S_SAFE_PATH.with_name("t2s_phrases.json").read_text(encoding="utf-8"))
    T2S_PHRASE_MAX = max(len(k) for k in T2S_PHRASES)
except Exception:
    T2S_PHRASES, T2S_PHRASE_MAX = {}, 0

# 轉換的目的是讓每個字都對應到一個完整 token。這份清單讓我們在轉完之後
# 還能指出哪些字沒救到——多半是台文專用字（佮、拚、囝），簡體字表也沒收。
try:
    T2S_FRAGMENTS = set(json.loads(
        T2S_SAFE_PATH.with_name("t2s_fragments.json").read_text(encoding="utf-8")))
except Exception:
    T2S_FRAGMENTS = set()


def remaining_fragments(text: str) -> str:
    """回傳仍會被拆成位元組的字（去重、保持出現順序）。"""
    seen: dict[str, None] = {}
    for ch in text:
        if ch in T2S_FRAGMENTS:
            seen[ch] = None
    return "".join(seen)


def to_simplified(text: str, mode: str) -> tuple[str, int]:
    """回傳轉換後的文字與換掉的字數，讓呼叫端能說明做了什麼。

    safe 只換「會碎片、簡體完整、且不與語料中其他字合併」的字。
    full 是整句轉簡，修掉更多碎片（兇、颱、齣 也在內），代價是簡化時的多對一
    合併會一併套用——不過那些合併很多本來就是同音字（颱/台、睏/困、齣/出 在
    台語都同音），真正有風險的是 麵/面、後/后、乾/幹/干 這類。
    哪種比較好是聽出來的，不是推論出來的。
    """
    table = T2S_SAFE if mode == "safe" else T2S_FULL if mode == "full" else None
    if not table:
        return text, 0

    pieces, i = [], 0
    while i < len(text):
        hit = None
        for span in range(min(T2S_PHRASE_MAX, len(text) - i), 1, -1):
            chunk = text[i:i + span]
            if chunk in T2S_PHRASES:
                hit = chunk
                break
        if hit:
            pieces.append(T2S_PHRASES[hit])
            i += len(hit)
        else:
            pieces.append(text[i].translate(table))
            i += 1
    out = "".join(pieces)
    return out, sum(1 for a, b in zip(text, out) if a != b)


app = FastAPI(title="CosyVoice3 checkpoint testbench")

_lock = threading.Lock()
_loaded: tuple[str, str, str] | None = None
_model = None


def apply_seed(seed: int) -> int:
    """Pin the RNG so a run can be repeated.

    Both stochastic steps in inference draw from torch's global generator: the
    LLM samples speech tokens with multinomial, and flow matching starts from
    torch.randn_like. Seeding both the CPU and CUDA generators is therefore
    enough to make a given (checkpoints, text, prompt, seed) reproduce exactly.

    The automatic seed comes from secrets rather than the random module, whose
    global state CausalConditionalCFM.__init__ resets with set_all_random_seed(0)
    on every model load. Drawing from it returned the same number each time a
    variant was swapped.
    """
    if seed < 0:
        seed = secrets.randbelow(MAX_SEED)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def list_parts(part: str) -> list[str]:
    """Official first, then whatever checkpoints were dropped in for this part."""
    found = sorted(p.stem for p in (VARIANTS_DIR / part).glob("*.pt")) if (VARIANTS_DIR / part).is_dir() else []
    found = [name for name in found if not name.endswith(HIDDEN_SUFFIXES)]
    return [OFFICIAL] + found


def part_path(part: str, name: str) -> Path:
    if name == OFFICIAL:
        return BASE_DIR / f"{part}.pt"
    path = VARIANTS_DIR / part / f"{name}.pt"
    if not path.is_file():
        raise HTTPException(400, f"unknown {part} checkpoint: {name}")
    return path


def build_variant_dir(llm: str, flow: str, hift: str) -> Path:
    """Symlink the official model, then point the three .pt names at the picks."""
    target = WORK_DIR / f"{llm}__{flow}__{hift}"
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in BASE_DIR.iterdir():
        if entry.name in ("llm.pt", "flow.pt", "hift.pt"):
            continue
        (target / entry.name).symlink_to(entry)
    for part, name in zip(PARTS, (llm, flow, hift)):
        (target / f"{part}.pt").symlink_to(part_path(part, name))
    return target


def get_model(llm: str, flow: str, hift: str):
    global _loaded, _model
    combo = (llm, flow, hift)
    if _loaded == combo and _model is not None:
        return _model
    if _model is not None:
        log.info("releasing %s", _loaded)
        _model = None
        _loaded = None
        torch.cuda.empty_cache()
    variant_dir = build_variant_dir(*combo)
    log.info("loading llm=%s flow=%s hift=%s", llm, flow, hift)
    _model = AutoModel(model_dir=str(variant_dir))
    _loaded = combo
    return _model


@app.get("/api/variants")
def variants():
    prompts = sorted(p.name for p in PROMPTS_DIR.glob("*.wav")) if PROMPTS_DIR.is_dir() else []
    return {
        "parts": {part: list_parts(part) for part in PARTS},
        "notes": MODEL_NOTES,
        "prompts": prompts,
        "loaded": list(_loaded) if _loaded else None,
        "default_instruct": DEFAULT_INSTRUCT,
        "base_present": BASE_DIR.is_dir(),
        "t2s_entries": T2S_COUNT,
        "t2s_full_entries": T2S_FULL_COUNT,
        "max_prompt_sec": MAX_PROMPT_SEC,
        "min_prompt_sec": MIN_PROMPT_SEC,
    }


class Health(BaseModel):
    cuda: bool
    device: str | None
    loaded: list[str] | None


@app.get("/api/health", response_model=Health)
def health():
    return Health(
        cuda=torch.cuda.is_available(),
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        loaded=list(_loaded) if _loaded else None,
    )


@app.post("/api/tts")
def tts(
    text: str = Form(...),
    llm: str = Form(OFFICIAL),
    flow: str = Form(OFFICIAL),
    hift: str = Form(OFFICIAL),
    instruct: str = Form(DEFAULT_INSTRUCT),
    prompt_name: str = Form(""),
    speed: float = Form(1.0),
    text_frontend: bool = Form(False),
    simplify: str = Form("none"),
    seed: int = Form(-1),
    prompt_wav: UploadFile | None = File(None),
):
    if not BASE_DIR.is_dir():
        raise HTTPException(503, f"official model not found under {BASE_DIR}")
    if seed < -1 or seed > MAX_SEED:
        raise HTTPException(400, f"seed must be -1 (random) or 0..{MAX_SEED}, got {seed}")

    if simplify not in ("none", "safe", "full"):
        raise HTTPException(400, f"simplify must be none/safe/full, got {simplify!r}")
    text, swapped = to_simplified(text, simplify)
    unfixable = remaining_fragments(text)

    # Form(DEFAULT_INSTRUCT) only fires when the field is absent, but the page always
    # sends it — so an emptied textarea arrives as "" and CosyVoice3 then dies deep
    # inside the LLM thread on `assert 151646 in text`, surfacing as an unrelated
    # "Input and output sizes should be greater than 0". Normalise it here instead.
    instruct = instruct.strip()
    if not instruct:
        instruct = DEFAULT_INSTRUCT
        log.info("instruct was blank, using the default")
    elif END_OF_PROMPT not in instruct:
        instruct = f"{instruct}{END_OF_PROMPT}"
        log.info("appended %s to the instruction", END_OF_PROMPT)
    # The frontend re-reads the prompt three times (mel at 24k, speech token, and
    # speaker embedding at 16k), so it needs a seekable path — a BytesIO would be
    # empty after the first read.
    tmp_prompt: Path | None = None
    if prompt_wav is not None:
        tmp_prompt = WORK_DIR / f"upload-{os.getpid()}-{threading.get_ident()}.wav"
        tmp_prompt.write_bytes(prompt_wav.file.read())
        source = tmp_prompt
    elif prompt_name:
        source = PROMPTS_DIR / prompt_name
        if not source.is_file():
            raise HTTPException(400, f"unknown prompt: {prompt_name}")
    else:
        raise HTTPException(400, "provide prompt_wav or prompt_name")

    try:
        try:
            info = sf.info(str(source))
            duration = info.frames / info.samplerate
        except Exception:
            duration = None  # 讓模型自己去報，不要因為讀不到長度就擋下
        if duration is not None:
            if duration > MAX_PROMPT_SEC:
                raise HTTPException(
                    400,
                    f"參考音檔長度 {duration:.1f} 秒，超過上限 {MAX_PROMPT_SEC:.0f} 秒。"
                    f"請裁切到 {MAX_PROMPT_SEC:.0f} 秒以內（建議 3–10 秒）。",
                )
            if duration < MIN_PROMPT_SEC:
                raise HTTPException(
                    400, f"參考音檔只有 {duration:.2f} 秒，太短不足以描述音色，請用 3 秒以上。"
                )

        with _lock:
            model = get_model(llm, flow, hift)
            used_seed = apply_seed(seed)
            log.info(
                "tts seed=%d llm=%s flow=%s hift=%s prompt=%s speed=%s "
                "simplify=%s(%d字)%s text=%r",
                used_seed, llm, flow, hift, Path(source).name, speed,
                simplify, swapped,
                f" 仍是碎片[{unfixable}]" if unfixable else "", text[:60],
            )
            # CosyVoice3 runs the LLM in its own thread. When that thread raises, the
            # token queue just ends up empty and the failure resurfaces here as a
            # zero-width interpolate — which says nothing about the real cause. Map it
            # back to something actionable and point at the container log for the
            # original traceback.
            try:
                chunks = [
                    out["tts_speech"]
                    for out in model.inference_instruct2(
                        text, instruct, str(source), stream=False, speed=speed, text_frontend=text_frontend
                    )
                ]
            except torch.cuda.OutOfMemoryError as error:
                # The card is shared with other services, so the usable headroom moves
                # around. Report what is actually free instead of a bare traceback.
                free, total = torch.cuda.mem_get_info()
                torch.cuda.empty_cache()
                raise HTTPException(
                    503,
                    f"GPU 記憶體不足（可用 {free / 2**20:.0f} MiB / 共 {total / 2**20:.0f} MiB）。"
                    f"這張卡與其他服務共用，長文字需要較大的 KV cache。"
                    f"請縮短文字，或停掉不需要的服務後重試。",
                ) from error
            except RuntimeError as error:
                if "should be greater than 0" not in str(error):
                    raise
                raise HTTPException(
                    500,
                    "模型沒有產生任何語音 token，多半是 instruction 或輸入文字有問題。"
                    "完整錯誤請看 `docker logs cosyvoice3-testbench`。",
                ) from error
            if not chunks:
                raise HTTPException(500, "model produced no audio")
            audio = torch.concat(chunks, dim=1)
            sample_rate = model.sample_rate
    finally:
        if tmp_prompt is not None:
            tmp_prompt.unlink(missing_ok=True)

    buf = io.BytesIO()
    torchaudio.save(buf, audio, sample_rate, format="wav")
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type="audio/wav",
        headers={
            "X-Variant": f"llm={llm};flow={flow};hift={hift}",
            "X-Duration-Sec": f"{audio.shape[1] / sample_rate:.2f}",
            "X-Seed": str(used_seed),
            # 標頭只能放 latin-1，轉換後的文字要編碼才帶得過去
            "X-Simplify": f"{simplify};{swapped}",
            "X-Spoken-Text": quote(text),
            "X-Unfixable": quote(unfixable),
            "Access-Control-Expose-Headers":
                "X-Variant, X-Duration-Sec, X-Seed, X-Simplify, X-Spoken-Text, X-Unfixable",
        },
    )


@app.exception_handler(Exception)
async def unhandled(request, exc):
    """Surface the real message; the default 500 body says nothing useful."""
    log.exception("request failed")
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


@app.get("/api/prompt/{name}")
def prompt_audio(name: str):
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "no such prompt")
    return FileResponse(path, media_type="audio/wav")


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "50010")))
