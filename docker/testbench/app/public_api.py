"""The `/v1` TTS interface that CastAgent (and anything else) calls.

Shaped to `tts-api-spec.md`. Two things in that spec drive most of the design:

  * **No model selection in the request.** The caller sends text and a voice; which
    checkpoints run is this server's decision. That keeps the caller stable while
    experiments continue, and it is the only way `model_version` can mean anything.
  * **Synchronous, one complete file.** No task ids, no polling, no streaming. An
    utterance is one to three seconds; the round trip costs less than the
    bookkeeping would.

The test bench's own `/api/*` routes stay as they are — they exist to compare
checkpoints and change whenever an experiment does. Both go through the same
`render()`, so what is heard in the browser is what the API returns.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import quote
from typing import Callable, Literal

import torch
import torchaudio
from fastapi import APIRouter, Depends, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field


# ---------------------------------------------------------------- 音色

# /v1 對外提供哪些音色，定義在 gitignore 的資料檔裡，不在程式碼。
#
# 兩個理由。一是**部署相關**：換一台機器，prompts/ 底下的音檔就不一樣，寫死在
# 程式裡等於假設所有機器長一樣。二是**授權**：音色來自哪個錄音、能不能對外用，
# 是每個部署自己的事——先前這裡寫死了九個八點檔抽出來的音色，那些沒有授權，
# 一旦有人 clone 下來跑起來就直接對外露出去了。
#
# 所以種子是空的。沒有設定檔時 /v1 回報零個音色，那是安全的失敗——寧可不能用，
# 也不要預設提供不該提供的東西。試聽台不受影響，它直接掃 prompts/ 目錄。
#
# 設定檔格式（TAIGI_DATA/voices.json）：
#   [{"voice_id": "young-female-01", "prompt": "檔名.wav", "label": "顯示名稱",
#     "gender": "male|female|unknown", "age": "child|teen|young|adult|senior|unknown",
#     "desc": "說明"}]
_BUILTIN_VOICES: list[dict] = []


# 音色清單是**部署相關的資料**，不是程式邏輯——換一台機器，prompts/ 底下的音檔就
# 不一樣。所以放在 gitignore 的資料目錄裡，程式只負責讀。
#
# 第一次啟動時用上面那份內建清單當種子寫出來，之後就以檔案為準。加音色是編輯
# JSON，不是改程式；那些檔案是誰的聲音、有沒有授權，本來就不該進版控。
VOICES_JSON = Path(os.environ.get("TAIGI_DATA", "/data/taigi")) / "voices.json"


def _load_voices() -> list[dict]:
    try:
        rows = json.loads(VOICES_JSON.read_text(encoding="utf-8"))
        if isinstance(rows, list) and rows:
            return rows
    except (OSError, ValueError):
        pass
    try:                                   # 第一次啟動：把內建清單寫成種子
        VOICES_JSON.parent.mkdir(parents=True, exist_ok=True)
        VOICES_JSON.write_text(
            json.dumps(_BUILTIN_VOICES, ensure_ascii=False, indent=1), encoding="utf-8")
    except OSError:
        pass                               # 唯讀掛載就算了，用記憶體裡那份
    return _BUILTIN_VOICES


VOICES: list[dict] = _load_voices()


# /v1 跟著試聽台跑：用「目前常駐在顯存的那一組」，而不是寫死一組。這是刻意的
# 取捨——好處是在瀏覽器切一下 /v1 就跟著換，不必編 compose 重建容器，兩邊也不會
# 因為想要不同模型而互相把對方踢出顯存（實測互踢一次要 22.8 秒）。
#
# 代價要知道：有人在試聽台做 A/B 時，/v1 的輸出會跟著變，同一集的不同句子可能
# 由不同模型產生。`model_version` 仍然如實反映當下用的是誰，但它不再是穩定的
# 快取依據——快取有可能存到別人試聽時經過的模型。
#
# 下面這三個只是**還沒有任何模型載入時**的起始值。
# 預設值只在環境變數沒設時生效；正式的來源是 docker-compose 的 TTS_API_LLM。
# 寫 official 是刻意的——猜一個 checkpoint 名字，換機器時它多半不存在。
LLM = os.environ.get("TTS_API_LLM", "official")
FLOW = os.environ.get("TTS_API_FLOW", "official")
HIFT = os.environ.get("TTS_API_HIFT", "official")
# 繁體字在 Qwen 的詞彙表裡沒有共用字形就退化成 UTF-8 位元組，模型從未把它們
# 當成一個字。對外一律轉簡體字形（只換字形，不換詞彙），呼叫端看不到這一層。
SIMPLIFY = os.environ.get("TTS_API_SIMPLIFY", "full")

# 合成是序列化的，所以請求會排隊。原本上限 45 秒是照短句校準的（當時一句 3–8
# 秒），但長文一段就要兩分鐘以上——排在後面的請求還沒輪到就被拒絕，而它只要再
# 等一下就會成功。拒絕一個會成功的請求，比讓它多等更糟。
#
# 現在放到 10 分鐘，足夠讓幾個長文排在前面。呼叫端的 HTTP timeout 要比這個大，
# 否則它自己會先斷線。真正等不到才回 429，而 Retry-After 用實測的合成速度估。
MAX_WAIT_SEC = float(os.environ.get("TTS_API_MAX_WAIT_SEC", "600"))
# 上限從 300 提到 5000：先前那個 300 是假的限制——超過只要切段就處理得了，
# 而不切段的話 200 字就已經生出講不完的音檔。留一個上限是為了擋住整本書，
# 因為合成時間與字數成正比，而這張卡與其他服務共用。
MAX_TEXT_CHARS = int(os.environ.get("TTS_API_MAX_TEXT_CHARS", "5000"))
# 每段的字數上限，交給 render() 切。
CHUNK_CHARS = int(os.environ.get("TTS_API_CHUNK_CHARS", "80"))
# 逗號要不要當切點，用 server 的環境變數（COMMA_SPLIT），不做成請求參數：
# 呼叫端不該需要知道這件事，而同一批內容用不同切法會前後不一致。
# 要換就換伺服器設定，整批一起換。


# ---------------------------------------------------------------- schema

class Voice(BaseModel):
    voice_id: str = Field(description="呼叫 /v1/synthesize 時使用的識別碼",
                          examples=["taigi-male-01"])
    label: str = Field(description="前端下拉選單顯示的文字",
                       examples=["阿明（臺灣台語・男）"])
    gender: Literal["male", "female", "unknown"] = Field(
        description="只會是這三個值之一。呼叫端常拿它直接分池，所以值必須正確。")
    desc: str = Field(description="這個音色的說明", examples=["八點檔男聲，中年、口條穩"])
    age: Literal["child", "teen", "young", "adult", "senior", "unknown"] = Field(
        default="unknown",
        description=("年齡層。child 孩童／teen 少年少女／young 青年／adult 壯年／"
                     "senior 年長。標 unknown 的是真人戲劇錄音，沒有可靠的年齡標註"
                     "——不要猜，猜錯比不標更糟。"))
    language: str = Field(default="zh-TW", description="BCP-47 語言標籤")
    sample_url: str | None = Field(
        default=None,
        description="試聽音檔。這裡回傳的是該音色的參考音（wav），不是合成結果。",
        examples=["/v1/voices/taigi-male-01/sample"])


class ModelInfo(BaseModel):
    """實際掛在跑的是哪幾個 checkpoint。

    `model_version` 是指紋，換模型會變，但看不出換成了誰。要確認「現在跑的到底是
    哪一版」得另外去查容器環境變數或日誌，那不該是使用者的事。
    """
    llm: str = Field(description="語言模型 checkpoint", examples=["llm_v7_best"])
    flow: str = Field(description="flow matching checkpoint", examples=["official"])
    hift: str = Field(description="聲碼器 checkpoint", examples=["official"])
    simplify: str = Field(description="繁→簡字形轉換模式（伺服器端自動套用）",
                          examples=["full"])


class VoicesResponse(BaseModel):
    model_version: str = Field(
        description=(
            "目前掛載的模型組合的指紋。checkpoint 換掉或檔案變動時這個值會變，"
            "呼叫端應把它併進快取 key，模型更新後舊音檔才會自動失效。"),
        examples=["taigi-2026.08-3f9a1c22"])
    model: ModelInfo
    voices: list[Voice]


class SynthesizeRequest(BaseModel):
    text: str = Field(description=f"要合成的文字，上限 {MAX_TEXT_CHARS} 字。"
                                  "輸入華語中文即可，繁簡都收，轉換在伺服器端完成。"
                                  "超過一句的長文會自動分段合成再串接——模型只在"
                                  "30 秒以內的樣本上訓練過，不分段會生出講不完的音檔。",
                      examples=["今天我們來聊這個話題。"])
    voice_id: str = Field(description="/v1/voices 回傳的其中一個",
                          examples=["taigi-male-01"])
    format: Literal["mp3", "wav"] = Field(
        default="mp3", description="預設 mp3。整條 pipeline 都是 mp3，在這裡轉一次比呼叫端每句轉一次划算。")
    speed: float = Field(default=1.0, ge=0.5, le=2.0, description="語速倍率")
    seed: int | None = Field(
        default=None,
        description="固定亂數種子以重現同一段音檔。省略則每次隨機——同一句話的語調會不同。",
        examples=[1942676902])


class ErrorResponse(BaseModel):
    error: str = Field(description="機器判讀用的錯誤代碼", examples=["voice_not_found"])
    message: str = Field(description="給人看的說明", examples=["找不到語音 taigi-male-99"])


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str
    model: ModelInfo
    cuda: bool
    detail: str | None = None


# ---------------------------------------------------------------- 錯誤

# auto_error=False：讓缺 token 走我們自己的 {error, message}，而不是 FastAPI
# 預設的 {detail}。宣告成 security scheme 之後 Swagger 才會出現 Authorize 按鈕，
# 否則在 /docs 按 Try it out 一定 401，而畫面上沒有地方可以填 token。
bearer = HTTPBearer(auto_error=False, description="填入 TTS_API_KEY，不含 Bearer 字樣")


class ApiError(Exception):
    """`{error, message}` 形狀的錯誤，與試聽台的 `{detail}` 分開。"""

    def __init__(self, status: int, code: str, message: str, retry_after: int | None = None):
        self.status, self.code, self.message, self.retry_after = status, code, message, retry_after
        super().__init__(message)


async def api_error_handler(_request: Request, exc: ApiError) -> JSONResponse:
    headers = {"Retry-After": str(exc.retry_after)} if exc.retry_after is not None else None
    return JSONResponse(status_code=exc.status,
                        content={"error": exc.code, "message": exc.message},
                        headers=headers)


# ---------------------------------------------------------------- router

def create_router(
    *,
    render: Callable[..., object],
    current_model: Callable[[], tuple[str, str, str]],
    prompts_dir: Path,
    variant_path: Callable[[str, str], Path | None],
    to_simplified: Callable[[str, str], tuple[str, int]],
    instruct: str,
) -> APIRouter:
    by_id = {v["voice_id"]: v for v in VOICES}
    # 只留參考音檔真的存在的音色。列出一個叫不動的 voice_id，呼叫端要到合成
    # 失敗才會知道。
    available = {k: v for k, v in by_id.items() if (prompts_dir / v["prompt"]).is_file()}

    def _model_info() -> ModelInfo:
        # Pydantic 的 BaseModel 不收位置參數，要逐個具名。
        llm, flow, hift = current_model()
        return ModelInfo(llm=llm, flow=flow, hift=hift, simplify=SIMPLIFY)

    def model_version() -> str:
        """由當下常駐的 checkpoint 檔案算出，換檔或換模型都會變。"""
        h = hashlib.sha256()
        for part, name in zip(("llm", "flow", "hift"), current_model()):
            h.update(f"{part}={name};".encode())
            path = variant_path(part, name)
            if path is not None and path.exists():
                stat = path.stat()
                h.update(f"{stat.st_size}:{int(stat.st_mtime)};".encode())
        return f"taigi-2026.08-{h.hexdigest()[:8]}"

    def require_token(cred: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        # 這個服務綁在 0.0.0.0:50010，同網段連得到，而它背後那張 20 GB 的卡還跑著
        # serve-tts-1、indextts-service、sensevoice-service。無認證的呼叫端把 GPU
        # 佔滿，倒的不只是這個服務。所以預設要 token。
        #
        # 內網自用覺得多餘的話，設 TTS_API_AUTH=off 明確關掉——要關可以，但必須是
        # 特意去關，不能是忘了設 key 就自動變開放。
        if os.environ.get("TTS_API_AUTH", "").lower() == "off":
            return
        expected = os.environ.get("TTS_API_KEY", "")
        if not expected:
            raise ApiError(503, "auth_not_configured",
                           "伺服器未設定 TTS_API_KEY。要停用認證請設 TTS_API_AUTH=off。")
        got = (cred.credentials if cred else "").strip()
        if not got or not secrets.compare_digest(got, expected):
            raise ApiError(401, "unauthorized",
                           "Authorization header 缺少或不正確。"
                           "在 Swagger 請按右上角 Authorize 填入 token（不含 Bearer 字樣）。")

    # 合成本身序列化執行；這個號誌只用來決定「等多久之後回 429」。
    gate = threading.BoundedSemaphore(1)
    # 排隊人數與觀測到的每字合成秒數，用來估等待時間並寫進 429 的訊息。
    waiting = threading.Semaphore(0)
    stats = {"queued": 0, "sec_per_char": 0.26, "lock": threading.Lock()}

    router = APIRouter(prefix="/v1", tags=["v1"])

    @router.get("/health", response_model=HealthResponse,
                summary="存活檢查", dependencies=[])
    def health() -> HealthResponse:
        """不需要 token，給 compose healthcheck 與負載平衡器用。"""
        cuda = torch.cuda.is_available()
        return HealthResponse(
            status="ok" if cuda and available else "degraded",
            model_version=model_version(),
            model=_model_info(),
            cuda=cuda,
            detail=None if cuda and available else
            ("沒有可用的 CUDA 裝置" if not cuda else "找不到任何參考音檔"),
        )

    @router.get("/voices", response_model=VoicesResponse,
                summary="列出可用音色",
                responses={401: {"model": ErrorResponse}, 503: {"model": ErrorResponse}})
    def voices(_: None = Depends(require_token)) -> VoicesResponse:
        """回傳音色清單與 `model_version`。

        `model_version` 會在 checkpoint 換掉時改變，把它併進快取 key，模型更新後
        舊音檔就會自動失效——否則同樣的文字會永遠命中舊模型產生的快取。
        """
        return VoicesResponse(
            model_version=model_version(),
            model=_model_info(),
            voices=[Voice(**{k: v for k, v in item.items() if k != "prompt"},
                          sample_url=f"/v1/voices/{item['voice_id']}/sample")
                    for item in available.values()],
        )

    @router.get("/voices/{voice_id}/sample",
                summary="音色試聽",
                response_class=FileResponse,
                responses={200: {"content": {"audio/wav": {}}},
                           404: {"model": ErrorResponse}})
    def sample(voice_id: str, _: None = Depends(require_token)):
        """該音色的參考音檔。這是輸入給模型的音色來源，不是合成結果。"""
        item = available.get(voice_id)
        if item is None:
            raise ApiError(404, "voice_not_found", f"找不到語音 {voice_id}")
        return FileResponse(prompts_dir / item["prompt"], media_type="audio/wav")

    @router.post(
        "/synthesize",
        summary="合成單句",
        response_class=Response,
        responses={
            200: {"content": {"audio/mpeg": {}, "audio/wav": {}},
                  "description": "二進位音檔。回應標頭帶 `X-Model-Version`、`X-Seed`、"
                                 "`X-Duration-Sec`；`X-Unfixable` 列出仍會念錯的字（多為台文專用字）。"},
            400: {"model": ErrorResponse, "description": "文字空白／過長，或 voice_id 不存在"},
            401: {"model": ErrorResponse},
            429: {"model": ErrorResponse, "description": "伺服器忙碌，依 `Retry-After` 重試"},
            503: {"model": ErrorResponse, "description": "顯存不足或模型未就緒"},
        },
    )
    def synthesize(req: SynthesizeRequest, _: None = Depends(require_token)):
        """把一句話合成成台語語音，同步回傳完整音檔。

        輸入華語中文即可，**不需要先轉成台文**。繁簡字形轉換在伺服器端自動完成，
        呼叫端不必知道，也不會改動任何一個詞——只換字形，不換用詞。
        """
        text = req.text.strip()
        if not text:
            raise ApiError(400, "empty_text", "text 不可為空白。")
        if len(text) > MAX_TEXT_CHARS:
            raise ApiError(400, "text_too_long",
                           f"text 有 {len(text)} 字，上限 {MAX_TEXT_CHARS} 字。"
                           f"長文會自動分段，但合成時間與字數成正比，"
                           f"而這張卡與其他服務共用。")
        item = available.get(req.voice_id)
        if item is None:
            raise ApiError(400, "voice_not_found",
                           f"找不到語音 {req.voice_id}。可用的有："
                           f"{', '.join(available) or '（無）'}")

        spoken, _swapped = to_simplified(text, SIMPLIFY)

        with stats["lock"]:
            stats["queued"] += 1
            ahead = stats["queued"] - 1
            estimate = int(ahead * stats["sec_per_char"] * len(spoken)) or 1
        try:
            if not gate.acquire(timeout=MAX_WAIT_SEC):
                raise ApiError(
                    429, "busy",
                    f"排隊超過 {MAX_WAIT_SEC:.0f} 秒仍未輪到，前面還有 {ahead} 個請求。"
                    f"合成是序列化的，這張卡與其他服務共用。",
                    retry_after=max(5, min(estimate, 300)))
        except ApiError:
            with stats["lock"]:
                stats["queued"] -= 1
            raise
        started = time.monotonic()
        try:
            out = render(
                spoken, llm=None, flow=None, hift=None, instruct=instruct,
                max_chunk_chars=CHUNK_CHARS,
                source=prompts_dir / item["prompt"], speed=req.speed,
                text_frontend=False, seed=-1 if req.seed is None else req.seed,
                note=f"v1 voice={req.voice_id}",
            )
            audio, sample_rate, used_seed = out.audio, out.sample_rate, out.seed
        finally:
            gate.release()
            with stats["lock"]:
                stats["queued"] -= 1
                # 用實際觀測值取代預設，估算才會跟著模型與文長變化。
                elapsed = time.monotonic() - started
                if spoken:
                    stats["sec_per_char"] = (stats["sec_per_char"] * 0.7
                                             + (elapsed / len(spoken)) * 0.3)

        buf = io.BytesIO()
        torchaudio.save(buf, audio, sample_rate, format=req.format)
        payload = buf.getvalue()
        from server import remaining_fragments  # 延後匯入，避免匯入時循環

        return Response(
            content=payload,
            media_type="audio/mpeg" if req.format == "mp3" else "audio/wav",
            headers={
                "X-Model-Version": model_version(),
                "X-Model-Llm": out.llm,
                "X-Seed": str(used_seed),
                "X-Duration-Sec": f"{audio.shape[1] / sample_rate:.2f}",
                "X-Render-Sec": f"{time.monotonic() - started:.2f}",
                # HTTP 標頭只能是 latin-1，而這裡放的是中文字（台文專用字，
                # 例：馀、佮、刣）。不編碼的話，一旦文字裡真的有碎片字就整個
                # 請求 500——先前測試的句子剛好都沒有，所以沒被發現。
                "X-Unfixable": quote(remaining_fragments(spoken)),
                "Access-Control-Expose-Headers":
                    "X-Model-Version, X-Model-Llm, X-Seed, X-Duration-Sec, "
                    "X-Render-Sec, X-Unfixable",
            },
        )

    return router
