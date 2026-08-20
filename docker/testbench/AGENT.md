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
  "model_version": "taigi-2026.08-c0b2231f",
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
| 文字上限 | 5000 字。**在句末標點處**自動分段（每段約 80 字）再串接 |
| 合成速度 | 短句 3–8 秒；長文約每字 0.26 秒（538 字約 140 秒）。第一次多 20 秒 |
| 併發 | 伺服器端序列化。`Semaphore(4)` 剛好——會排隊但不會失敗 |
| 沒有 batch | 模型常駐，省下的只有 HTTP 來回，之後真的變瓶頸再加 |

---

## 長文與它的一個限制

模型只在 30 秒以內的樣本上訓練過（所有語料的 `max-duration` 都是 30 秒），
整段丟進去會生出講不完的音檔。所以超過 80 字就**在句末標點處**（`。？！；：`）
自動分段，逐段合成再串接。538 字切成 7 段、音長 127 秒。

**但只有逗號的段落切不開。** 新聞稿常這樣寫——整段用逗號串連，句號只在最後一個字。
那種文字會維持一整段送進模型，超過 80 字就可能講不完，尾巴整句消失。

```
✅ 「⋯開始影響台灣東部海域。氣象專家提醒⋯」   有句號，切得開
⚠️ 「⋯開始影響台灣東部海域，氣象專家提醒⋯」   只有逗號，切不開
```

**呼叫端可以自己先斷句**，那是最可靠的做法：在語意完整的地方加句號，或直接把
長段落拆成多次呼叫。這樣分段點是你決定的，不是伺服器猜的。

伺服器端也能改成在逗號切（`COMMA_SPLIT=1`），但**預設關閉**：在逗號切會把句子
從中間剖開，前半被當成句尾唸完收尾、後半又從頭起一句，語意還在延續而語調已經
結束又重來，聽起來就是接不上。兩種失敗各有各的難聽，所以留給人決定。

日誌會記每一段的長度，太短會標警告——出問題時查得到是哪一段。

---

## 已知念不對的

不是 bug，是模型的能力邊界。知道了才不會把時間花在錯的地方。

| 類型 | 例子 | 能不能繞過 |
|---|---|---|
| **華語詞 ≠ 台語詞** | 東西 → 應念 mih8-kiann7（物件），模型逐字念成 tang-si | **可以**，附上台羅標音。辭典查得到這個對應 |
| 語料裡沒有的字 | 坤、廬（「逆轉乾坤」「三顧茅廬」） | 部分可以，見下 |
| 破音字 | 乾坤的「乾」該念 khiân 不是 kan | **可以**，附上台羅標音 |

台羅標音的用法——**整句換成台羅，或整句都不換**：

```
他準備 gik8-tsuan2 khian5-khun1。
```

實測同一句、五個 seed：不標音 2/3 念對，標音 5/5。標音壓得過模型自己的偏好。

**不要用括號並列的寫法。** 寫成「乾坤(khian5-khun1)」模型會**念兩遍**——先念漢字
再念羅馬字。要覆寫就得把原字換掉，並列等於什麼都沒覆寫還多講一次。

**也不要只換句子裡的幾個詞。** 漢字夾在台羅中間是唯一會出事的形態：量過 10 個
seed，混排 1/10 會幻覺（3.5 秒的句子生出 21.6 秒），而純漢字與全台羅都是 0/10。

```
✅ 純漢字        記得把東西放進冰箱。                              0/10 幻覺
✅ 整句台羅      e7-ki3-tit4 ka7 mih8-kiann7 khng3--jip8-khi3⋯   0/10
❌ 混排          e7-ki3-tit4 把 mih8-kiann7 te2--jip8 冰箱。      1/10
❌ 括號並列      他準備逆轉乾坤(khian5-khun1)。                   「乾坤」念兩次
```

要整句換的話 `tools/taigi_annotate.py` 會做，但**它挑讀音挑得很差**（把→lak8、
冰箱→ping1-kui7 冰櫃）。辭典查得到讀音，查不到這一句該用哪一個——那需要看
上下文。挑錯的輸出很流利，錯得不明顯，比幻覺更難察覺。

讀音查得到的話（教育部辭典、台華線頂）就能標；查不到的字（例：廬）**寧可不標，
不要猜**——編造的讀音比不標更糟。

---

## 檢查是否活著

```bash
curl $TTS_BASE_URL/v1/health     # 不需要 token
```

```json
{
  "status": "ok",
  "model_version": "taigi-2026.08-c0b2231f",
  "model": {"llm": "llm_v7_best", "flow": "official",
            "hift": "official", "simplify": "full"},
  "cuda": true,
  "detail": null
}
```

`model` 寫出實際掛的 checkpoint。目前是 **v7**（v6 底 + 媠聲 4.62 h），
是聽完全部候選之後選的——它的 CV loss 在候選裡最差，而 loss 最低的那個連 155 字
都講不完。**它會跟著試聽台變**——有人在瀏覽器裡切換模型，
`/v1` 就跟著用那一個，`model_version` 也跟著變。這是刻意的取捨：換模型不必重建
容器，代價是 agent 在跑的時候若有人在做 A/B，輸出會跟著換。

`status` 是 `degraded` 時 `detail` 會說明原因（沒有 CUDA、找不到參考音檔）。
