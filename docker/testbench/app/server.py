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

import contextlib
import io
import json
import logging
import os
import secrets
import shutil
import sys
import threading
import time
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
# vLLM 加速 LLM 那一段。實測 RTF 從 1.35 降到 0.39（LLM 10.33s → 2.01s，快 5.1 倍）。
#
# 預設關閉，兩個原因：一是它要 torch 2.7 + vllm 0.9 的環境（見 docker/vllm-test），
# 正式映像跑的是 torch 2.3.1；二是它第一次載入要多花約 3 分鐘匯出權重，而試聽台
# 的用途是一直換模型比較——那個成本在對外服務上划算，在研究工具上不划算。
LOAD_VLLM = os.environ.get("LOAD_VLLM", "0") not in ("0", "", "false", "False")

# flow decoder 走 TensorRT。只開 vLLM 的話 LLM 快了 5 倍，但 flow 沒動——它的
# 佔比就從 15% 變成 37%，變成新的瓶頸。社群拆解的數字也是聲碼器約 32%。
#
# 第一次要把 ONNX 編譯成 TRT 引擎（十幾分鐘），之後沿用。需要
# flow.decoder.estimator.fp32.onnx 在模型目錄裡，用 cosyvoice/bin/export_onnx.py 產生。
LOAD_TRT = os.environ.get("LOAD_TRT", "0") not in ("0", "", "false", "False")

# 同一個 seed 為什麼不是每次都一樣。量到的是純 torch 六次相同、vLLM 也是六次
# 相同——**兩邊一樣**，所以不是 vLLM 造成的，這件事本來就在。
#
# 剩下的解釋是 cuDNN 會依實測時間挑卷積演算法、cuBLAS 的歸約順序不固定，
# 加上 TF32。這些都讓同樣的輸入算出末位不同的結果，而 CFM 是迭代解 ODE，
# 早期一點點差異會被放大成完全不同的一遍朗讀。
#
# 這組旗標把上面幾件事全部釘死，代價是速度。要嚴謹比 seed 時才開。
# 注意 cuBLAS 那個要用環境變數，而且必須在第一次 CUDA 呼叫之前設好。
# 分兩級是因為兩者的代價差很多，見下面的實測。
#   1 / light —— 只釘 cuDNN 的演算法挑選與 TF32
#   full      —— 再加 use_deterministic_algorithms，強迫每個運算都用決定性核心
# 可以逗號組合，用來分辨是哪一項在收速度的錢：
#   tf32   關掉 TF32，矩陣乘法走完整 fp32
#   cudnn  cudnn.deterministic，強迫卷積用決定性演算法
#   algo   use_deterministic_algorithms，範圍最廣
#   full   等於 tf32,cudnn,algo
DETERMINISTIC = os.environ.get("TTS_DETERMINISTIC", "0").lower()
_det = set(DETERMINISTIC.replace("full", "tf32,cudnn,algo").split(","))
if _det & {"tf32", "cudnn", "algo"}:
    torch.backends.cudnn.benchmark = False
    if "tf32" in _det:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if "cudnn" in _det:
        torch.backends.cudnn.deterministic = True
    if "algo" in _det:
        # 只有 use_deterministic_algorithms 需要這個，單開 cudnn 不必——把它綁在
        # 一起會分不清速度是被誰吃掉的
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only：有些運算沒有決定性版本，寧可讓它跑完並警告，也不要整個掛掉
        torch.use_deterministic_algorithms(True, warn_only=True)
    log.info("決定性模式 %s 已啟用", sorted(_det & {"tf32", "cudnn", "algo"}))

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

CONCURRENCY = max(1, int(os.environ.get("TTS_CONCURRENCY", "2")))
TRT_CONCURRENT = max(1, int(os.environ.get("TRT_CONCURRENT", "1")))


class Gate:
    """Admit several requests at once, but let some of them demand the GPU alone.

    Everything under CosyVoice2Model is already built to be shared: session state
    is keyed by uuid, the vLLM engine batches whatever requests are in flight, and
    the TensorRT decoder hands out execution contexts from a pool. Serialising
    calls was our own doing, not the model's.

    What is not shareable is the seed. Flow matching draws its starting noise from
    torch's *global* generator, so two runs overlapping in time interleave their
    draws and neither reproduces what it would have produced alone. That is fine
    when the caller asked for a random seed and only wants audio, and not fine on
    a page whose whole purpose is comparing one seed against another. So a pinned
    seed takes the gate exclusively; everything else shares up to `slots`.

    Model swaps are exclusive for the obvious reason: the weights move underneath
    anyone mid-inference.
    """

    def __init__(self, slots: int):
        self.slots = slots
        self._cv = threading.Condition()
        self._shared = 0          # 目前共享名額用掉幾個
        self._exclusive = False   # 有沒有人正在獨佔
        self._waiting_ex = 0      # 等著獨佔的人數，用來擋住後來的共享請求

    @contextlib.contextmanager
    def hold(self, exclusive: bool):
        with self._cv:
            if exclusive:
                self._waiting_ex += 1
                # 等到場子完全清空
                self._cv.wait_for(lambda: not self._exclusive and self._shared == 0)
                self._waiting_ex -= 1
                self._exclusive = True
            else:
                # 有人在等獨佔就先讓，否則長串的共享請求會把它餓死
                self._cv.wait_for(
                    lambda: not self._exclusive and not self._waiting_ex
                    and self._shared < self.slots)
                self._shared += 1
        try:
            yield
        finally:
            with self._cv:
                if exclusive:
                    self._exclusive = False
                else:
                    self._shared -= 1
                self._cv.notify_all()


_gate = Gate(CONCURRENCY)
_lock = threading.Lock()          # 只保護模型切換，不再包住整段合成
_loaded: tuple[str, str, str] | None = None
_model = None


_vllm_seed: int | None = None

# 補上 seed 之後，同一個 seed 八次仍有兩次不同。剩下的來源不是 RNG：flow 的起始
# 噪音是載入時就算好的常數（CausalConditionalCFM.rand_noise），所以變異全在 LLM
# 採樣——bf16 kernel 加上 CUDA graph 重播，logits 末位會浮動，偶爾把 top-k 的
# 選擇翻掉。enforce_eager 關掉 CUDA graph 換取穩定，代價是解碼變慢。
# 用環境變數控制，預設關閉：多數時候要的是速度，比 seed 的時候才需要開。
VLLM_EAGER = os.environ.get("VLLM_EAGER", "0") not in ("0", "", "false", "False")

# 診斷用：貪婪解碼（temperature=0）把採樣的隨機性整個拿掉。如果連貪婪都還會
# 每次不同，那變異就不在 RNG 而在 kernel 的數值，補 seed 是補不到的。
VLLM_GREEDY = os.environ.get("VLLM_GREEDY", "0") not in ("0", "", "false", "False")

# vLLM 的 KV 池佔顯存的比例。上游寫死 0.2，留空就沿用。設了就覆蓋——
# 理由見 _patch_engine_args 的說明：那裡有一百倍的餘裕，讓給 TRT 比較划算。
VLLM_GPU_FRACTION = float(os.environ.get("VLLM_GPU_FRACTION", "0") or 0) or None

# 一條序列最長幾個 token。上游沿用 Qwen 的 32768，但 CosyVoice 的序列短得多：
# max_len = (文字長度 - prompt 長度) x max_token_text_ratio，80 字上限也才 1600。
#
# 這個值不只是上限，vLLM 會拿它檢查「KV 池至少要放得下一整條最長序列」。
# 32768 配上調低的 gpu_memory_utilization 就會直接拒絕啟動——它建議調高記憶體，
# 但正確的做法是把這個不切實際的長度改成實際會用到的。
VLLM_MAX_LEN = int(os.environ.get("VLLM_MAX_LEN", "0") or 0) or None


def _patch_engine_args(vllm) -> None:
    """調整 vLLM 的引擎參數，不動上游檔案。

    cosyvoice/cli/model.py 把 gpu_memory_utilization 寫死成 0.2。實測那是浪費：
    KV 池子拿到 2.06 GiB，一個 token 佔 12 KiB，等於放得下 18 萬個 token；而
    一發 7.5 秒的請求只用約 308 個，同時放得下 585 發。實際的併發天花板是 4，
    卡在 flow 與聲碼器的算力，六發併發期間 preemption 是 0 次。

    也就是說這 2 GiB 有一百倍的餘裕。把它讓出來給 TensorRT 的第二個執行上下文，
    flow 那段才有機會真的並行——那才是現在擋住併發的地方。
    """
    if getattr(vllm.EngineArgs, "_taigi_wrapped", False):
        return
    original = vllm.EngineArgs

    def patched(*args, **kwargs):
        if VLLM_EAGER:
            kwargs.setdefault("enforce_eager", True)
        if VLLM_GPU_FRACTION:
            kwargs["gpu_memory_utilization"] = VLLM_GPU_FRACTION
        if VLLM_MAX_LEN:
            kwargs["max_model_len"] = VLLM_MAX_LEN
        return original(*args, **kwargs)

    patched._taigi_wrapped = True
    vllm.EngineArgs = patched


def _pin_vllm_seed() -> None:
    """Make torch's seed reach vLLM's sampler, which otherwise ignores it.

    Without vLLM both stochastic steps draw from torch's global generator, so
    torch.manual_seed pins the whole run. vLLM samples inside its own engine
    instead, and cosyvoice/llm/llm.py builds its SamplingParams without a seed —
    so the engine's generator just keeps advancing from request to request. The
    symptom is quiet and easy to miss: the same seed still returns audio, still
    returns the *right* audio, but a different take each time. Measured here as
    three distinct outputs across six identical requests.

    SamplingParams is imported inside inference_wrapper, so replacing the name on
    the vllm module is enough and no upstream file has to be edited. Only pinned
    seeds inject a value, and those hold the gate exclusively — hence a plain
    global rather than anything thread-aware. Random-seed requests leave vLLM's
    default alone, which is what makes them safe to overlap.
    """
    try:
        import vllm
    except ImportError:
        return                                  # 沒開 vLLM，torch 的 seed 本來就管用
    _patch_engine_args(vllm)
    if getattr(vllm.SamplingParams, "_taigi_wrapped", False):
        return
    original = vllm.SamplingParams

    def seeded(*args, **kwargs):
        if _vllm_seed is not None and kwargs.get("seed") is None:
            kwargs["seed"] = _vllm_seed
        if VLLM_GREEDY:
            kwargs["temperature"] = 0.0
        return original(*args, **kwargs)

    seeded._taigi_wrapped = True
    vllm.SamplingParams = seeded


def apply_seed(seed: int) -> int:
    """Pin the RNG so a run can be repeated.

    Both stochastic steps in inference draw from torch's global generator: the
    LLM samples speech tokens with multinomial, so seeding both the CPU and CUDA
    generators pins a given (checkpoints, text, prompt, seed). Under vLLM that
    half needs _pin_vllm_seed() as well.

    This used to claim flow matching contributes randomness too, via
    torch.randn_like. It does not: CosyVoice3 uses CausalConditionalCFM, whose
    __init__ precomputes self.rand_noise once under set_all_random_seed(0) and
    slices the same block on every call. The seed has only ever controlled token
    sampling — worth knowing when a run varies and the LLM looks innocent.

    The automatic seed comes from secrets rather than the random module, whose
    global state CausalConditionalCFM.__init__ resets with set_all_random_seed(0)
    on every model load. Drawing from it returned the same number each time a
    variant was swapped.
    """
    global _vllm_seed
    pinned = seed >= 0
    if seed < 0:
        seed = secrets.randbelow(MAX_SEED)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if LOAD_VLLM:
        _pin_vllm_seed()
        _vllm_seed = seed if pinned else None
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
    # vLLM 會把權重匯出到 {model_dir}/vllm，那要花約 3 分鐘。每次都 rmtree 重建
    # 等於每換一次模型就重跑一次匯出，而試聽台的用途就是一直換。已經備好的就沿用。
    if target.exists():
        if LOAD_VLLM and (target / "vllm").is_dir():
            _link_trt_artifacts(target, flow)   # 沿用的目錄也要接上 TRT 快取
            return target
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in BASE_DIR.iterdir():
        if entry.name in ("llm.pt", "flow.pt", "hift.pt"):
            continue
        (target / entry.name).symlink_to(entry)
    for part, name in zip(PARTS, (llm, flow, hift)):
        (target / f"{part}.pt").symlink_to(part_path(part, name))
    _link_trt_artifacts(target, flow)
    return target


# TRT 的 ONNX 與編譯好的引擎放這裡，而不是變體目錄裡面。上面那個 rmtree 是為了
# 換模型時乾淨重建，它不知道自己順手刪掉的是要花十幾分鐘才生得出來的東西——
# 這件事已經發生過一次，服務因此 500。
#
# 依 flow 的名字分開存：估計器的權重來自 flow.pt，換了 flow 就是另一個引擎。
TRT_CACHE = WORK_DIR / "_trt"
TRT_FILES = ("flow.decoder.estimator.fp32.onnx",
             "flow.decoder.estimator.fp32.mygpu.plan")


def _link_trt_artifacts(target: Path, flow: str) -> None:
    """把變體目錄裡的 TRT 檔名指到快取。

    引擎還沒編譯過的時候故意留下懸空連結：convert_onnx_to_trt 用 open(path,"wb")
    寫檔，會跟著連結寫到快取那一端，下次就直接沿用。
    """
    if not LOAD_TRT:
        return
    cache = TRT_CACHE / flow
    cache.mkdir(parents=True, exist_ok=True)
    for name in TRT_FILES:
        link = target / name
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(cache / name)


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
    # trt_concurrent 是 TensorRT 執行上下文的池子大小。理想上要跟 TTS_CONCURRENCY
    # 一樣多，flow 解碼才不會變成新的隊伍——但這張卡開不到 2：另一個服務佔著
    # 5.3 GB，vLLM 又預留 0.2 的顯存，第二個上下文直接建不起來（common.py:206
    # 的 assert）。留在 1 沒有正確性問題，池子的 queue.get() 會安全地排隊，
    # 只是 flow 那一段仍然一次一個。顯存變寬裕再調這個環境變數。
    if LOAD_VLLM:
        _pin_vllm_seed()      # EngineArgs 在載入時就用到，補丁要趕在這之前
    _model = AutoModel(model_dir=str(variant_dir), load_vllm=LOAD_VLLM,
                       load_trt=LOAD_TRT, trt_concurrent=TRT_CONCURRENT)
    _loaded = combo
    return _model


def render_stream(
    text: str,
    *,
    llm: str | None,
    flow: str | None,
    hift: str | None,
    instruct: str,
    source: Path,
    speed: float,
    seed: int,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
    comma_split: bool = COMMA_SPLIT,
):
    """逐塊產出音訊，而不是等整段合成完。

    yield (音訊張量, 取樣率, 這是第幾塊, 從請求開始算的秒數)。

    跟 render() 的差別只有 stream=True——模型內部改成一邊生 speech token
    一邊送進 flow，湊夠一個 hop 就吐一塊。代價是每塊之間要重疊處理，總時間
    比一次做完略長；換到的是**第一個音出來的時間**大幅縮短。

    併發規則跟 render() 一樣走 _gate。串流會把閘佔得比較久（要等整段送完），
    所以指定 seed 的串流請求會明顯擋住別人——這是重現性的代價，不是 bug。
    """
    t0 = time.perf_counter()
    exclusive = seed >= 0
    with _gate.hold(exclusive):
        with _lock:
            if llm is None or flow is None or hift is None:
                resident = _loaded or startup_model()
                llm = llm if llm is not None else resident[0]
                flow = flow if flow is not None else resident[1]
                hift = hift if hift is not None else resident[2]
            model = get_model(llm, flow, hift)
            used_seed = apply_seed(seed)
        segments = split_for_synthesis(text, max_chunk_chars, comma_split)
        log.info("stream seed=%d llm=%s prompt=%s 段數=%d text=%r",
                 used_seed, llm, source.name, len(segments), text[:60])
        index = 0
        for piece in segments:
            for out in model.inference_instruct2(
                    piece, instruct, str(source), stream=True,
                    speed=speed, text_frontend=False):
                index += 1
                yield (out["tts_speech"], model.sample_rate, index,
                       time.perf_counter() - t0, used_seed)


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

    併發由 _gate 管：指定 seed 的獨佔（見 Gate 的說明），其餘共享名額。
    模型切換另外用 _lock 保護，因為權重會在別人推論到一半時被換掉。
    """
    exclusive = seed >= 0
    with _gate.hold(exclusive):
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
            # 音色名要回報。否則前端只能記自己送出時的下拉選單值，而那份會在
            # 合成期間被使用者換掉——卡片就標成「現在選的」而不是「這段實際用的」。
            "X-Prompt-Name": quote(prompt_name or ""),
            "X-Annotate": quote(annotate_note),
            "X-Annotated-Text": quote(text[:900]),
            "X-Annotate-Changed": quote(json.dumps(annotate_changed, ensure_ascii=False)[:1800]),
            "X-Spoken-Text": quote(text),
            "X-Unfixable": quote(unfixable),
            # 沒列進來的 header 瀏覽器讀不到，前端就會安靜地退回預設值。
            # 加新 header 時這裡要一起加——先前 X-Annotate 那組就漏過。
            "Access-Control-Expose-Headers": ", ".join((
                "X-Variant", "X-Duration-Sec", "X-Seed", "X-Simplify",
                "X-Spoken-Text", "X-Unfixable", "X-Prompt-Name",
                "X-Annotate", "X-Annotated-Text", "X-Annotate-Changed",
            )),
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


@app.post("/api/tts/stream")
def tts_stream(
    text: str = Form(...),
    llm: str = Form(OFFICIAL),
    flow: str = Form(OFFICIAL),
    hift: str = Form(OFFICIAL),
    instruct: str = Form(DEFAULT_INSTRUCT),
    prompt_name: str = Form(""),
    speed: float = Form(1.0),
    simplify: str = Form("none"),
    annotate: str = Form("none"),
    hanji_table: str = Form(""),
    max_chunk_chars: int = Form(MAX_CHUNK_CHARS),
    seed: int = Form(-1),
):
    """串流合成。每塊音訊即時送出，不等整段做完。

    格式是一行 JSON 後面接一塊 WAV，重複到結束：

        {"chunk":1,"bytes":41036,"at":1.83,"sec":0.85}\n<wav bytes>
        {"chunk":2,...}\n<wav bytes>
        {"done":true,"chunks":9,"total_sec":17.16,"elapsed":12.4,"seed":7}\n

    `at` 是從請求開始算到這塊送出的秒數——第一塊的 at 就是「多久聽得到聲音」，
    那是串流唯一真正買到的東西（總時間反而會略長）。

    用自訂格式而不是 multipart：呼叫端要邊收邊播，multipart 的解析器多半會
    先把整份收完，那就失去意義了。
    """
    if not BASE_DIR.is_dir():
        raise HTTPException(503, f"official model not found under {BASE_DIR}")
    if annotate not in annotate_layer.MODES:
        raise HTTPException(400, f"annotate must be one of {annotate_layer.MODES}")
    source = PROMPTS_DIR / prompt_name
    if not prompt_name or not source.is_file():
        raise HTTPException(400, f"unknown prompt: {prompt_name!r}")

    body, _note, _changed = annotate_layer.apply(text, annotate, hanji_table or None)
    if not annotate.startswith("hanji"):
        body, _ = to_simplified(body, simplify)

    def gen():
        try:
            total = 0.0
            last = (0, 0.0, seed)
            for audio, sr, index, at, used_seed in render_stream(
                    body, llm=llm, flow=flow, hift=hift, instruct=instruct,
                    source=source, speed=speed, seed=seed,
                    max_chunk_chars=max_chunk_chars):
                buf = io.BytesIO()
                torchaudio.save(buf, audio, sr, format="wav")
                data = buf.getvalue()
                sec = audio.shape[1] / sr
                total += sec
                last = (index, total, used_seed)
                head = json.dumps({"chunk": index, "bytes": len(data),
                                   "at": round(at, 3), "sec": round(sec, 3)},
                                  ensure_ascii=False)
                yield head.encode() + b"\n" + data
            yield (json.dumps({"done": True, "chunks": last[0],
                               "total_sec": round(last[1], 2),
                               "seed": last[2], "spoken": body[:900]},
                              ensure_ascii=False) + "\n").encode()
        except Exception as exc:                      # 串流開始後才失敗的話，
            log.exception("stream failed")            # HTTP 狀態碼已經送出去了，
            yield (json.dumps({"error": str(exc)}) + "\n").encode()   # 只能寫進內容

    return StreamingResponse(gen(), media_type="application/octet-stream",
                             headers={"X-Variant": f"llm={llm};flow={flow};hift={hift}",
                                      "Cache-Control": "no-store"})


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
    # /v1 的入場號誌要跟 Gate 一致，否則放行了也擠不進來
    concurrency=CONCURRENCY,
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

@app.get("/stream", include_in_schema=False)
def stream_page():
    """/stream 是串流試聽頁。StaticFiles 只認 .html，補一條乾淨的路徑。"""
    return FileResponse(STATIC_DIR / "stream.html",
                        headers={"Cache-Control": "no-cache, must-revalidate"})


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
