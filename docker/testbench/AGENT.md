# Agent 串接指南

怎麼從程式呼叫這個台語 TTS。完整的端點規格看 [API.md](API.md)，
互動式文件在 <http://localhost:50010/docs>。

本文的程式碼都實際跑過。

---

## 最重要的一件事

**直接送華語中文,不要先翻成台文。**

```python
await tts.synthesize("你到底在哪裡？我找你很久了。", voice_id, path)   # ✅
```

模型訓練時看的就是華語書寫的文字,翻譯層只會讓句子偏離原意——先前用外部
華語→台文 API 的做法就是這樣壞掉的,它會改寫整句。

繁體簡體都收,字形轉換在伺服器端做,你不會看到,也不會有任何一個詞被換掉
（只換字形,不換用詞）。

---

## 兩行設定

```bash
export TTS_BASE_URL=http://localhost:50010
export TTS_API_KEY=<向部署者索取；在 docker/testbench/.env>
```

沒帶 token 一律 401；伺服器沒設 token 則 `/v1` 整個停用回 503。

---

## 兩個呼叫

### 列出音色

```bash
curl -H "Authorization: Bearer $TTS_API_KEY" $TTS_BASE_URL/v1/voices
```

```json
{
  "model_version": "taigi-2026.08-1cf89d92",
  "voices": [
    {"voice_id": "taigi-male-01", "label": "男聲 01（臺灣台語）", "gender": "male",
     "desc": "中低音 110 Hz，抑揚明顯，語速偏慢。八點檔對白",
     "language": "zh-TW", "sample_url": "/v1/voices/taigi-male-01/sample"}
  ]
}
```

目前 **5 男 5 女**,`gender` 都是人耳確認過的,可以直接拿來分池。

### 合成

```bash
curl -X POST $TTS_BASE_URL/v1/synthesize \
  -H "Authorization: Bearer $TTS_API_KEY" -H "Content-Type: application/json" \
  -d '{"text":"你到底佇佗位？","voice_id":"taigi-male-01"}' -o out.mp3
```

同步回完整的 mp3（24 kHz 單聲道）。沒有 task id、不用輪詢、不是串流。

---

## 可以直接用的 driver

滿足 CastAgent 的 `TTSDriver` Protocol：

```python
"""CastAgent TTSDriver —— 對接自建台語 TTS。"""
from __future__ import annotations
import asyncio, os
from pathlib import Path
import httpx

BASE = os.environ.get("TTS_BASE_URL", "http://localhost:50010")
KEY = os.environ["TTS_API_KEY"]


class TaigiTTS:
    def __init__(self, timeout: float = 900.0):
        self._client = httpx.AsyncClient(
            base_url=BASE,
            headers={"Authorization": f"Bearer {KEY}"},
            timeout=timeout,
        )

    async def voices(self) -> dict:
        r = await self._client.get("/v1/voices")
        r.raise_for_status()
        return r.json()

    async def synthesize(self, text: str, voice_id: str, out_path: Path) -> None:
        for attempt in range(3):
            r = await self._client.post("/v1/synthesize",
                                        json={"text": text, "voice_id": voice_id})
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                return
            if r.status_code == 429 and attempt < 2:
                await asyncio.sleep(float(r.headers.get("Retry-After", 5)))
                continue
            if r.status_code in (500, 503) and attempt < 2:
                await asyncio.sleep(2 ** attempt)
                continue
            body = r.json()
            raise RuntimeError(f"{body.get('error')}: {body.get('message')}")
        raise RuntimeError("重試三次仍然失敗")

    async def aclose(self) -> None:
        await self._client.aclose()
```

`timeout` 給 900 秒是因為三件事疊起來：第一次請求要多花約 20 秒載入模型；長文會
分段合成，538 字約 140 秒；而合成是序列化的，排在別人後面還要再等。實測一個長文
加三個短句同時送，最後一個在 218 秒完成——**全部成功，沒有一個被拒絕**。

伺服器端等到 600 秒才回 429。**呼叫端的 timeout 必須比它大**，否則客戶端先斷線，
而伺服器還在幫它合成，白燒 GPU。

### 分池與併發

```python
cat = await tts.voices()
pool = {"male": [], "female": [], "unknown": []}
for v in cat["voices"]:
    pool[v["gender"]].append(v["voice_id"])
# → {'male': 5, 'female': 5, 'unknown': 0}

sem = asyncio.Semaphore(4)

async def one(role, voice_id, text, path):
    async with sem:
        await tts.synthesize(text, voice_id, path)

await asyncio.gather(*(one(*x) for x in lines))
```

`Semaphore(4)` 可以維持。伺服器端合成是序列化的（顯示卡與其他服務共用，並行只會
互搶顯存），四個併發會依序完成、全部成功，不會失敗。實測 3.4 / 6.7 / 10.1 / 13.0 秒。

---

## 快取

快取 key **一定要併進 `model_version`**：

```python
key = hashlib.md5(f"{model_version}|{voice_id}|{text}".encode()).hexdigest()
```

換 checkpoint 時 `model_version` 會變，舊音檔自動失效。少了它，同一句話會永遠命中
舊模型產生的快取，而模型已經換過了。

`model_version` 從 `/v1/voices` 拿，或從合成回應的 `X-Model-Version` 標頭拿。

---

## 錯誤

一律 `{"error": "...", "message": "..."}`。

| 狀態 | `error` | 該怎麼做 |
|---|---|---|
| 400 | `empty_text` / `text_too_long` / `voice_not_found` | **不要重試**，參數有問題 |
| 401 | `unauthorized` | 檢查 token |
| 429 | `busy` | 依 `Retry-After` 重試。等到 600 秒才會出現，訊息會說前面還有幾個 |
| 500 / 503 | — | 退避重試；連續失敗就插靜音，不要中斷整輪 |

`voice_not_found` 的 message 會列出所有可用的 voice_id，除錯時直接看得到。

---

## 回應標頭

```
X-Model-Version   算快取用
X-Seed            這次用的種子。要重現同一段音檔就把它放進請求的 seed
X-Duration-Sec    音檔長度
X-Render-Sec      合成花的時間
X-Unfixable       仍會念錯的字
```

`X-Unfixable` 值得記進 log。那是台文專用字（佮、刣、囡），簡體也沒有對應字形，
模型看到的是位元組碎片。累積下來就知道哪些字需要補資料。

---

## 選用參數

```json
{"text": "...", "voice_id": "...", "format": "mp3", "speed": 1.0, "seed": 1942676902}
```

| 欄位 | 預設 | 說明 |
|---|---|---|
| `format` | `mp3` | 也可 `wav`。整條 pipeline 是 mp3 的話不用改 |
| `speed` | `1.0` | 0.5–2.0 |
| `seed` | 隨機 | 固定以重現。**同一句話不給 seed，每次語調會不同** |

沒有 `pitch`——CosyVoice3 的推論介面沒有這個參數。

---

## 限制

| | |
|---|---|
| 文字上限 | 5000 字。超過一句會自動分段（每段約 80 字）再串接 |
| 合成速度 | 短句 3–8 秒；長文約每字 0.26 秒（538 字約 140 秒）。第一次多 20 秒 |
| 併發 | 伺服器端序列化。`Semaphore(4)` 剛好——會排隊但不會失敗 |
| 沒有 batch | 模型常駐，省下的只有 HTTP 來回，之後真的變瓶頸再加 |

---

## 檢查是否活著

```bash
curl $TTS_BASE_URL/v1/health     # 不需要 token
```

```json
{
  "status": "ok",
  "model_version": "taigi-2026.08-1cf89d92",
  "model": {"llm": "llm_v5_suisiann_best", "flow": "official",
            "hift": "official", "simplify": "full"},
  "cuda": true,
  "detail": null
}
```

`model` 寫出實際掛的 checkpoint。**它會跟著試聽台變**——有人在瀏覽器裡切換模型，
`/v1` 就跟著用那一個，`model_version` 也跟著變。這是刻意的取捨：換模型不必重建
容器，代價是 agent 在跑的時候若有人在做 A/B，輸出會跟著換。

`status` 是 `degraded` 時 `detail` 會說明原因（沒有 CUDA、找不到參考音檔）。
