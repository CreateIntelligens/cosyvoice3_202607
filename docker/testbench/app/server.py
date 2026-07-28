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
import logging
import os
import secrets
import shutil
import sys
import threading
from pathlib import Path

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
# torch accepts a much wider range, but 32-bit keeps the value short enough to
# quote in a bug report and matches what the UI offers.
MAX_SEED = 2 ** 31 - 1
# frontend.py asserts this before extracting speech tokens; catching it here turns
# an opaque 500 into something the caller can act on.
MAX_PROMPT_SEC = 30.0
MIN_PROMPT_SEC = 1.0
DEFAULT_INSTRUCT = "You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>"

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
        "prompts": prompts,
        "loaded": list(_loaded) if _loaded else None,
        "default_instruct": DEFAULT_INSTRUCT,
        "base_present": BASE_DIR.is_dir(),
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
    seed: int = Form(-1),
    prompt_wav: UploadFile | None = File(None),
):
    if not BASE_DIR.is_dir():
        raise HTTPException(503, f"official model not found under {BASE_DIR}")
    if seed < -1 or seed > MAX_SEED:
        raise HTTPException(400, f"seed must be -1 (random) or 0..{MAX_SEED}, got {seed}")
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
                "tts seed=%d llm=%s flow=%s hift=%s prompt=%s speed=%s text=%r",
                used_seed, llm, flow, hift, Path(source).name, speed, text[:60],
            )
            chunks = [
                out["tts_speech"]
                for out in model.inference_instruct2(
                    text, instruct, str(source), stream=False, speed=speed, text_frontend=text_frontend
                )
            ]
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
            "Access-Control-Expose-Headers": "X-Variant, X-Duration-Sec, X-Seed",
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
