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
from typing import NamedTuple
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
from cosyvoice.utils.frontend_utils import split_paragraph  # noqa: E402

import public_api  # noqa: E402  對外 /v1 介面，見 public_api.py

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
    # 時數以 v2 workspace 的 prepared/summary.json 為準：train 255.30h，
    # 連 dev/test 合計 284.47h。先前這裡寫 324h 是錯的。
    "llm_v2_best": "v2 · tai8+naer 255h(train) · 目前基準",
    "llm_v2_avg5": "v2 · tai8+naer 255h(train) · 平均",
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
    # 這裡原本標「驗證用，非成品」，是訓練前寫下的預期——4.62 小時單一語者，
    # 本來以為漢字那側會災難性遺忘。聽測結果相反：漢字沒退步，破音字反而變準，
    # 判定比 v2 好一點。標籤已經不成立，改掉。
    "llm_v5_suisiann_best": "v5 · v2+媠聲4.6h · 會讀台羅標音 · 判定優於 v2",
    "llm_v5_suisiann_avg5": "v5 · v2+媠聲 · 平均",
    # v6 與 v2 只差資料量、與 v4 只差 naer 用台文還是華語，三者互為單一變因對照。
    # naer 全數 107,292 句都在，v4 因為文字轉換漏了 6,250 句。
    "llm_v6_best": "v6 · 新 tai8+naer台文 291h(train) · 對照 v2(資料量)、v4(文字)",
    "llm_v6_avg5": "v6 · 新 tai8+naer台文 · 平均",
    "llm_v6_ep3": "v6 · epoch 3 原始檔（loss 最低，等同 best）",
    # v7 = v6 + 媠聲，第二階段的設定與 v5 完全相同。兩者的 loss 曲線幾乎重疊
    # （v5 ep3 2.647 / v7 ep3 2.644），代表這一步是可重現的，不是碰巧。
    "llm_v7_best": "v7 · v6+媠聲4.6h · 資料與台羅標音都到位",
    # v8（+教育部 33 h）與 v9（+naer 197 h）已從試聽台移除。放大台羅第二階段
    # 是失敗的：從第 2 個 epoch 起就漏字、重複整句、把「東西」念成 tang-si 而
    # 不是物件，而 CV loss 一路在降——v9b_ep19 的 loss 最低（1.352 對 v7 的
    # 2.644）、accuracy 最高，155 字卻只講得出 1.8 秒。
    # 完整紀錄見 reports/taigi-tailo-scaling-2026-08-20.md。
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
# 所有語料的 max-duration 都設 30 秒，模型從來沒見過更長的樣本。一段 200 字要講
# 一分鐘，遠在訓練分布之外——LLM 的自迴歸生成到後面失去依據，產出的 speech token
# 變成雜訊，而且會提早停：實測 211 字不切段只生出 41 秒，切段後 67 秒才是對的。
#
# CosyVoice 自己的 text_frontend 會切，但它同時做數字正規化、去括號、刪除漢字與
# 拉丁字母之間的空格。那些對實驗對照是變數，對台羅標音也可能有影響，所以這裡只
# 借它的切段函式，正規化不啟用。
# 一段最多幾個字。實測字/秒：60 字 3.09、80 字 3.18、110 字 3.89（尾巴已經在掉）。
MAX_CHUNK_CHARS = int(os.environ.get("MAX_CHUNK_CHARS", "80"))

# 逗號要不要也當切點。預設關閉，因為在逗號切會把句子從中間剖開：前半被當成句尾
# 唸完收尾，後半又從頭起一句，語意還在延續而語調已經結束又重來——聽起來就是
# 接不上。在句號切就沒有這個問題。
#
# 代價是整段只有逗號的文字（新聞稿很常見）切不開，超過上限就會講不完、尾巴整句
# 消失。兩種失敗各有各的難聽，所以做成開關讓人選，而不是替人決定。
COMMA_SPLIT = os.environ.get("COMMA_SPLIT", "0") not in ("0", "", "false", "False")

# frontend.py asserts this before extracting speech tokens; catching it here turns
# an opaque 500 into something the caller can act on.
MAX_PROMPT_SEC = 30.0
MIN_PROMPT_SEC = 1.0
# CosyVoice3 tokenises this to id 151646 and asserts on its presence; without it the
# LLM thread dies before emitting a single token.
END_OF_PROMPT = "<|endofprompt|>"
DEFAULT_INSTRUCT = f"You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。{END_OF_PROMPT}"


def startup_model() -> tuple[str, str, str]:
    """還沒有任何模型載入時要用哪一組。

    render() 與 /v1 的回報端必須共用這一個答案。先前 render() 退回 official 而
    回報端退回環境變數，結果 health 說跑的是 v5、實際合成用的是官方模型——
    音檔和標頭對不起來，而且沒有任何錯誤。
    """
    return (public_api.LLM, public_api.FLOW, public_api.HIFT)

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


import annotate_layer  # noqa: E402

app = FastAPI(
    title="CosyVoice3 台語 TTS",
    version="1.0.0",
    description=(
        "兩組介面共用同一個模型與同一條合成路徑。\n\n"
        "- **`/v1/*`** —— 給程式呼叫的穩定介面。只收文字與音色，模型由伺服器端決定。\n"
        "- **`/api/*`** —— 試聽台自己用的，可以任意組合 checkpoint，隨實驗變動，不保證相容。\n\n"
        "兩者走同一個 `render()`，所以聽到的和 agent 拿到的是同一個東西。"
    ),
    openapi_tags=[
        {"name": "v1", "description": "對外 TTS 介面。需要 Bearer token。"},
        {"name": "testbench", "description": "試聽台內部介面，會隨實驗變動。"},
    ],
)

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
        # 頁面的預設選擇。跟 /v1 掛的是同一組，換版本只要改 .env，
        # 不必再回來改前端寫死的名字（先前寫死 llm_v2_best，早就過期了）。
        "startup": dict(zip(PARTS, startup_model())),
    }


def split_for_synthesis(text: str, max_chars: int, comma_split: bool) -> list[str]:
    """把長文切成模型吃得下的段落。短文原樣回傳，不動它。

    先在句末標點切，那是最自然的停頓。但一段話可能整段只有逗號、句號只在最後
    一個字——新聞稿常常這樣寫——那樣切完仍然是一整段。實測 155 字的氣象稿就是
    這種情形：切成「1/1 段」，模型講到一半停掉，聽起來就是後面某段整個消失。
    所以仍然過長的段落再用逗號切一次。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    def by_punctuation(chunk: str, commas: bool) -> list[str]:  # noqa: D401
        return [p for p in split_paragraph(
            chunk, lambda t: t, "zh",
            token_max_n=max_chars, token_min_n=max(20, max_chars * 3 // 4),
            merge_len=20, comma_split=commas) if p.strip()]

    pieces: list[str] = []
    for piece in by_punctuation(text, False) or [text]:
        if len(piece) <= max_chars or not comma_split:
            # 關閉逗號切時，過長的段落原樣送出——模型可能講不完，但至少不會
            # 在句子中間斷開。日誌會把「太短」標出來，聽的人知道發生了什麼。
            pieces.append(piece)
            continue
        finer = by_punctuation(piece, True)
        # 連逗號都沒有就只能硬切。寧可在詞中間斷，也不要整段講不完。
        if not finer or max(len(p) for p in finer) > max_chars:
            finer = [piece[i:i + max_chars] for i in range(0, len(piece), max_chars)]
        pieces.extend(finer)
    return pieces or [text]


class Rendered(NamedTuple):
    audio: torch.Tensor
    sample_rate: int
    seed: int
    llm: str
    flow: str
    hift: str


def render(
    text: str,
    *,
    llm: str | None,
    flow: str | None,
    hift: str | None,
    instruct: str,
    source: Path,
    speed: float,
    text_frontend: bool,
    seed: int,
    note: str = "",
    max_chunk_chars: int = 0,
    comma_split: bool = COMMA_SPLIT,
) -> Rendered:
    """Synthesise one utterance.

    Both the test bench and the public API come through here. They diverged once
    before — batch_eval.py converted text locally while the server converted it
    again — and the two produced different audio for what looked like the same
    request. One path means that cannot happen.

    Passing None for the three checkpoints means "whatever is already resident",
    which is how /v1 follows the test bench. That choice has to be made inside the
    lock: read it outside and a switch between the read and the synthesis would
    have us report one model while using another.

    Holds the GPU lock for the whole call: the three modules together need several
    GB and this card is shared, so two concurrent syntheses would fight over VRAM
    rather than overlap usefully.
    """
    with _lock:
        if llm is None or flow is None or hift is None:
            resident = _loaded or startup_model()
            llm = llm if llm is not None else resident[0]
            flow = flow if flow is not None else resident[1]
            hift = hift if hift is not None else resident[2]
        model = get_model(llm, flow, hift)
        used_seed = apply_seed(seed)
        # 種子只設一次，之後 RNG 自然往下走——與上游逐段合成的行為一致。
        segments = split_for_synthesis(text, max_chunk_chars, comma_split)
        log.info(
            "render seed=%d llm=%s flow=%s hift=%s prompt=%s speed=%s %s "
            "段數=%d text=%r",
            used_seed, llm, flow, hift, source.name, speed, note,
            len(segments), text[:60],
        )
        # CosyVoice3 runs the LLM in its own thread. When that thread raises, the
        # token queue just ends up empty and the failure resurfaces here as a
        # zero-width interpolate — which says nothing about the real cause. Map it
        # back to something actionable and point at the container log for the
        # original traceback.
        try:
            # 逐段收，而不是一個列表推導：一段沒產出時要看得見。先前寫成推導，
            # 失敗的段落只是不貢獻元素——音檔少了一截、總長變短，而日誌上什麼
            # 都沒有。使用者聽到「中間某段沒聲音」「最後一段直接不見」就是這個。
            chunks = []
            empty: list[int] = []
            for index, piece in enumerate(segments, 1):
                got = [out["tts_speech"] for out in model.inference_instruct2(
                    piece, instruct, str(source), stream=False,
                    speed=speed, text_frontend=text_frontend,
                )]
                produced = sum(c.shape[1] for c in got)
                seconds = produced / model.sample_rate
                # 每段都記時長。空段落固然要抓，但「有聲音卻明顯太短」同樣是
                # 失敗——那是模型講到一半停掉，聽起來就是某一段消失了。
                # 實測這個模型約每秒 3.5–4.5 字。門檻先前設每秒 8 字太寬鬆——
                # 155 字只生出 27.8 秒（每秒 5.6 字）明顯是講不完，卻沒被抓到。
                expected = len(piece) / 5.5
                if not got or produced == 0:
                    empty.append(index)
                    log.warning("第 %d/%d 段沒有產出任何語音：%r",
                                index, len(segments), piece[:40])
                elif seconds < expected:
                    empty.append(index)
                    log.warning("第 %d/%d 段只有 %.1f 秒但有 %d 字（至少該有 %.1f 秒），"
                                "多半是講到一半停掉：%r",
                                index, len(segments), seconds, len(piece),
                                expected, piece[:40])
                else:
                    log.info("  第 %d/%d 段 %.1f 秒 / %d 字",
                             index, len(segments), seconds, len(piece))
                chunks.extend(got)
            if empty:
                log.warning("共 %d/%d 段沒有聲音（第 %s 段），輸出會比預期短",
                            len(empty), len(segments),
                            "、".join(str(i) for i in empty))
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
        return Rendered(torch.concat(chunks, dim=1), model.sample_rate,
                        used_seed, llm, flow, hift)


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
    # 標音層：none 原字／corpus 語料學出的讀音／gemini 多義詞問 Gemini。
    # 整句換或整句不換，不做混排——混排 1/10 會幻覺。
    annotate: str = Form("none"),
    # 台文對照表用哪一版。v5 是目前的（對齊法，乾淨但覆蓋窄），v1 是辭典法
    # （覆蓋廣但噪音多）。留著讓耳朵直接比，見 taigi_store.BASES。
    hanji_table: str = Form(""),
    # 這一次請求專用的替換，JSON {"華語詞":"台語漢字"}。用來試聽「改了會怎樣」，
    # 不寫進任何一層——聽完滿意才用 /admin 真的寫進去。
    preview_override: str = Form(""),
    # 長文自動分段。預設開啟——關掉的話 200 字以上就會生出一段講不完的音檔。
    # 要逐字對照的實驗才關（設 0）。
    max_chunk_chars: int = Form(MAX_CHUNK_CHARS),
    # 逗號切預設照環境變數，但試聽台可以逐次覆寫，方便直接 A/B 兩種切法。
    comma_split: bool = Form(COMMA_SPLIT),
    seed: int = Form(-1),
    prompt_wav: UploadFile | None = File(None),
):
    if not BASE_DIR.is_dir():
        raise HTTPException(503, f"official model not found under {BASE_DIR}")
    if seed < -1 or seed > MAX_SEED:
        raise HTTPException(400, f"seed must be -1 (random) or 0..{MAX_SEED}, got {seed}")

    if simplify not in ("none", "safe", "full"):
        raise HTTPException(400, f"simplify must be none/safe/full, got {simplify!r}")
    if annotate not in annotate_layer.MODES:
        raise HTTPException(400, f"annotate must be one of {annotate_layer.MODES}, got {annotate!r}")
    # 標音要在繁簡轉換之前：辭典的鍵是繁體，而換完之後剩下的漢字才需要轉簡。
    try:
        extra = json.loads(preview_override) if preview_override else None
    except json.JSONDecodeError:
        raise HTTPException(400, "preview_override 不是合法 JSON")
    text, annotate_note, annotate_changed = annotate_layer.apply(
        text, annotate, hanji_table or None, extra)
    # 台文模式之後不能再繁簡轉換。繁簡轉換是為華語輸入設計的（避免罕見繁體字被
    # 拆成位元組），但台語漢字轉簡體會被換成讀音不同的華語字——「連紲」變「连绁」，
    # 紲 lian5-sua3 變成 绁 xiè，模型就唸成 lian5-sia。台羅模式沒事，因為那不是漢字。
    if annotate.startswith("hanji"):
        simplify, swapped = "none", 0
    else:
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

        out = render(
            text, llm=llm, flow=flow, hift=hift, instruct=instruct, source=Path(source),
            speed=speed, text_frontend=text_frontend, seed=seed,
            max_chunk_chars=0 if text_frontend else max_chunk_chars,
            comma_split=comma_split,
            note=f"simplify={simplify}({swapped}字) annotate={annotate_note}"
                 + (f" 仍是碎片[{unfixable}]" if unfixable else ""),
        )
        audio, sample_rate, used_seed = out.audio, out.sample_rate, out.seed
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
            "X-Annotate": quote(annotate_note),
            "X-Annotated-Text": quote(text[:900]),
            "X-Annotate-Changed": quote(json.dumps(annotate_changed, ensure_ascii=False)[:1800]),
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


class Feedback(BaseModel):
    verdict: str                      # up 或 down
    text: str                         # 原始華語
    sent: str                         # 實際送進模型的文字
    mode: str = "none"
    changed: list = []                # [[華語詞, 台語漢字], ...]
    blamed: list[str] = []            # 使用者指出哪幾個詞不對；空的就是未歸因
    suggested: dict[str, str] = {}    # {華語詞: 正確寫法}，選填


@app.get("/api/dicts")
def dicts(q: str = "", layer: str = "", limit: int = 200, offset: int = 0,
          version: str = ""):
    """瀏覽對照表。唯讀，不需要權杖——內容不敏感，**改**它才需要（/admin）。

    version 切底表（v1/v3/v4/v5）；不給就是目前生效的那份。人工修正與 LLM
    補的那兩層對每個版本都生效，所以切版本只換底表。
    """
    import sys as _sys
    _sys.path.insert(0, "/opt/CosyVoice/tools")
    from taigi_store import BASES, layers
    if version and version not in BASES:
        raise HTTPException(404, f"沒有這個版本，可用的是 {sorted(BASES)}")
    base, llm, over = layers(version or None)

    def unwrap(v):
        return v["hanji"] if isinstance(v, dict) else v

    rows = []
    for w in set(base) | set(llm) | set(over):
        src = "override" if w in over else ("llm" if w in llm else "table")
        if layer and src != layer:
            continue
        eff = unwrap(over.get(w)) or unwrap(llm.get(w)) or base.get(w)
        if q and q not in w and q not in (eff or ""):
            continue
        rows.append({"word": w, "hanji": eff, "source": src,
                     "table": base.get(w), "llm": unwrap(llm.get(w)),
                     "override": unwrap(over.get(w)),
                     "why": (over.get(w) or {}).get("why", "") if isinstance(over.get(w), dict) else ""})
    # 人工修正排前面，那是最該被看見與複查的
    rows.sort(key=lambda r: ({"override": 0, "llm": 1, "table": 2}[r["source"]], r["word"]))
    import sys as _sys2
    _sys2.path.insert(0, "/opt/CosyVoice/tools")
    from taigi_store import BASES as _B
    return {"total": len(rows),
            "counts": {"table": len(base), "llm": len(llm), "override": len(over)},
            "version": version or "目前生效", "versions": sorted(_B),
            "rows": rows[offset:offset + limit]}


@app.get("/api/suggestions")
def suggestions():
    """看得到自己填過什麼。唯讀、不需要金鑰——這是意見，不是設定。

    同一個詞被建議多次就合併，次數多的排前面；已經套用進表的標出來，
    才分得清「還沒處理」和「處理過了」。
    """
    import sys as _sys
    _sys.path.insert(0, "/opt/CosyVoice/tools")
    from taigi_store import FEEDBACK, dismissed, merged
    now, gone = merged(), dismissed()
    agg: dict = {}
    down_only: dict = {}
    try:
        for line in FEEDBACK.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if row.get("verdict") != "down":
                continue
            # 台文與台羅改的是不同的表，混在一起會套錯地方——「氣壓」在 hanji
            # 被點 5 次、在 corpus 被點 2 次，那是兩件不同的事。
            kind = "hanji" if str(row.get("mode", "")).startswith("hanji") else "tailo"
            for w, h in (row.get("suggested") or {}).items():
                k = (w, h, kind)
                e = agg.setdefault(k, {"word": w, "suggest": h, "kind": kind,
                                       "times": 0, "samples": []})
                e["times"] += 1
                if len(e["samples"]) < 3 and row.get("text"):
                    e["samples"].append(row["text"][:40])
            for w in row.get("blamed", []):
                if not (row.get("suggested") or {}).get(w):
                    d = down_only.setdefault((w, kind),
                                             {"word": w, "kind": kind, "times": 0, "samples": []})
                    d["times"] += 1
                    if len(d["samples"]) < 3 and row.get("text"):
                        d["samples"].append(row["text"][:40])
    except OSError:
        pass
    for e in agg.values():
        # 台羅的建議套不進台文表，標出來讓人知道那要另外處理
        e["applicable"] = e["kind"] == "hanji"
        e["current"] = now.get(e["word"])
        e["applied"] = now.get(e["word"]) == e["suggest"]
        e["nochange"] = e["suggest"] == e["word"]
        e["dismissed"] = f"{e['word']}→{e['suggest']}" in gone
        e["todo"] = not (e["applied"] or e["dismissed"] or e["nochange"])
    for d in down_only.values():
        d["current"] = now.get(d["word"])
        d["applicable"] = d["kind"] == "hanji"
    # 排除過的沉到最後，跟已套用的一樣算「處理完了」
    rank = sorted(agg.values(), key=lambda e: (not e["todo"], -e["times"]))
    return {"suggested": rank,
            "todo": sum(1 for e in rank if e["todo"]),
            "blamed_only": sorted(down_only.values(), key=lambda d: -d["times"])}


@app.post("/api/feedback")
def feedback(fb: Feedback):
    """點讚點爛。

    整句生成、一個詞爛就整句爛——所以未歸因的負評也要收：同一個詞出現在很多
    爛句子、很少好句子，統計上就浮得出來。blamed 只是加速這個過程。
    """
    if fb.verdict not in ("up", "down"):
        raise HTTPException(400, "verdict must be up or down")
    import sys as _sys
    _sys.path.insert(0, "/opt/CosyVoice/tools")
    from taigi_store import log_feedback
    ok = log_feedback(verdict=fb.verdict, text=fb.text, sent=fb.sent,
                      mode=fb.mode, changed=fb.changed, blamed=fb.blamed,
                      suggested=fb.suggested)
    if not ok:
        raise HTTPException(503, "回饋寫不進去，檢查 TAIGI_DATA 掛載")
    return {"ok": True}


@app.get("/api/prompt/{name}")
def prompt_audio(name: str):
    path = PROMPTS_DIR / name
    if not path.is_file():
        raise HTTPException(404, "no such prompt")
    return FileResponse(path, media_type="audio/wav")


def _variant_path(part: str, name: str) -> Path | None:
    """part_path 但不會丟例外——算 model_version 時不該因為檔案不在就 500。"""
    try:
        return part_path(part, name)
    except HTTPException:
        return None


def _current_model() -> tuple[str, str, str]:
    """/v1 跟著試聽台：回報常駐的那一組，還沒載入就回起始值。"""
    return _loaded or startup_model()


app.include_router(public_api.create_router(
    render=render,
    current_model=_current_model,
    prompts_dir=PROMPTS_DIR,
    variant_path=_variant_path,
    to_simplified=to_simplified,
    instruct=DEFAULT_INSTRUCT,
))
app.add_exception_handler(public_api.ApiError, public_api.api_error_handler)

# /admin 改的是全站共用的對照表，權杖跟 /v1 分開——合成的權限不該順便帶著
# 「改變所有人發音」的權限。沒設 ADMIN_API_KEY 就整組不掛載（不是 401，是不存在）。
import admin_api  # noqa: E402
if admin_api.ADMIN_KEY:
    app.include_router(admin_api.router)
else:
    log.info("ADMIN_API_KEY 未設定，/admin 維護端點不掛載")

@app.get("/dicts", include_in_schema=False)
@app.get("/dicts/{version}", include_in_schema=False)
def dicts_page(version: str = ""):
    """/dicts 是目前生效的表，/dicts/v1 之類是指定底表版本。

    StaticFiles 只認 /dicts.html，這裡補乾淨的路徑；版本由頁面自己從網址讀。
    """
    # 這頁還在頻繁改，瀏覽器快取住舊版會讓人以為功能不見了（實際發生過：
    # 加了「我的意見」頁籤，/dicts/v5 卻還顯示舊版）。明講不要快取。
    return FileResponse(STATIC_DIR / "dicts.html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})


# 這一行要放在所有路由之後：掛在 "/" 的 StaticFiles 會接住剩下的一切。
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "50010")))
