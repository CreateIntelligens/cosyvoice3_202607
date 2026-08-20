# `/v1` TTS API

給程式呼叫的介面，依 `tts-api-spec.md` 實作。互動式文件在
**<http://localhost:50010/docs>**（Swagger UI），機器可讀的規格在 `/openapi.json`。

## 兩組介面的分工

| 前綴 | 給誰用 | 穩定性 |
|---|---|---|
| `/v1/*` | agent、其他服務 | **穩定**。只收文字與音色，模型由伺服器決定 |
| `/api/*` | 試聽台網頁 | 隨實驗變動，可任意組合 checkpoint，不保證相容 |

兩者共用同一個 `render()`，所以瀏覽器裡聽到的和 agent 拿到的是同一個東西。
先前 `batch_eval.py` 自己做過一次文字轉換、伺服器又做一次，同樣的請求產出不同
音檔——一條路徑就不會有這種事。

## 認證

```
Authorization: Bearer <TTS_API_KEY>
```

### token 放哪

`docker/testbench/.env`，首次啟動時自動產生，權限 600，已在 `.gitignore` 裡。

```bash
grep TTS_API_KEY docker/testbench/.env
```

**改完 `.env` 要用 `docker compose up -d`，不能用 `docker restart`。** 環境變數是
建立容器時烘進去的，`restart` 只是重啟同一個容器，改動不會生效——實測改成別的值
再 restart，容器裡仍是舊 token，舊 token 照樣通得過。

（`app/` 底下的程式碼是 bind mount，那個 `restart` 就會生效，兩者不一樣。）

`docker compose up -d` 會比對設定，變了才重建；沒變就顯示 `Running` 不動它。

### 要不要認證

這個服務綁 `0.0.0.0:50010`，同網段連得到（實測 `http://10.9.0.35:50010` 通），
而它背後那張 20 GB 的卡還跑著 `serve-tts-1`、`indextts-service`、
`sensevoice-service`。無認證的呼叫端把 GPU 佔滿，倒的不只是這個服務。

內網自用覺得多餘的話：

```yaml
TTS_API_AUTH: "off"     # 在 docker-compose.yaml 的 environment
```

要關可以，但必須是特意去關。**忘記設 `TTS_API_KEY` 不會變成開放**，而是回 503
停用——那種預設比較難出事。

`/v1/health` 一律不需要 token，方便 compose healthcheck 與負載平衡器。

## 端點

### `GET /v1/voices`

```json
{
  "model_version": "taigi-2026.08-c0b2231f",
  "voices": [
    {
      "voice_id": "taigi-male-01",
      "label": "阿明（臺灣台語・男）",
      "gender": "male",
      "desc": "八點檔男聲，中年、口條穩，適合旁白與敘事",
      "language": "zh-TW",
      "sample_url": "/v1/voices/taigi-male-01/sample"
    }
  ]
}
```

`gender` 只會是 `male` / `female` / `unknown`。呼叫端常拿它直接當 dict key 分池，
填錯不會報錯、會安靜地把男角配到女聲，所以檔名沒寫明的一律 `unknown`，不用名字猜。
確認之後在 `app/public_api.py` 的 `VOICES` 改。

`model_version` 由掛載的 checkpoint 檔案（名稱、大小、mtime）算出來。**請把它併進
快取 key**：模型換版之後舊音檔才會自動失效，否則同樣的文字會永遠命中舊模型的快取。

### `POST /v1/synthesize`

```bash
curl -X POST http://localhost:50010/v1/synthesize \
  -H "Authorization: Bearer $TTS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"text":"今天我們來聊這個話題。","voice_id":"taigi-male-01"}' \
  -o out.mp3
```

| 欄位 | 必要 | 預設 | 說明 |
|---|---|---|---|
| `text` | ✅ | — | 上限 5000 字。長文自動分段。**輸入華語中文即可**，繁簡都收 |
| `voice_id` | ✅ | — | `/v1/voices` 的其中一個 |
| `format` | ⬜ | `mp3` | `mp3` 或 `wav` |
| `speed` | ⬜ | `1.0` | 0.5–2.0 |
| `seed` | ⬜ | 隨機 | 固定以重現同一段音檔 |

同步回傳完整的二進位音檔（`audio/mpeg`，24 kHz 單聲道）。沒有 task id、沒有輪詢、
沒有串流。回應標頭：

```
X-Model-Version   算快取用
X-Seed            這次實際用的種子，要重現就帶回去
X-Duration-Sec    音檔長度
X-Render-Sec      合成花的時間
X-Unfixable       仍會念錯的字（多為台文專用字，簡體也沒有對應字形）
```

### `GET /v1/voices/{voice_id}/sample`

該音色的**參考音檔**（wav）。這是輸入給模型的音色來源，不是合成結果。

### `GET /v1/health`

不需 token，也是**確認現在跑哪個模型**最快的方法：

```bash
curl http://10.9.0.35:50010/v1/health
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

`model_version` 是指紋，換模型會變，但看不出換成了誰；`model` 才寫出實際掛的
checkpoint。合成的回應也帶 `X-Model-Llm`，所以拿到一個音檔之後還查得出它是誰產的。

## 錯誤

一律 `{"error": "<代碼>", "message": "<說明>"}`。

| 狀態 | `error` | 什麼時候 |
|---|---|---|
| 400 | `empty_text` / `text_too_long` / `voice_not_found` | 參數問題，重試沒用 |
| 401 | `unauthorized` | token 缺少或不正確 |
| 429 | `busy` | 佇列等待超過 45 秒。**帶 `Retry-After`** |
| 503 | `auth_not_configured` | 伺服器沒設 `TTS_API_KEY` |
| 503 | — | 顯存不足（訊息會寫出當下可用多少） |

## 效能與併發

合成是**序列化**的：三個模組加起來要好幾 GB 顯存，而這張 20 GB 的卡與
`serve-tts-1`、`indextts-service`、`sensevoice-service` 共用，並行只會互相搶記憶體。

實測（暖機後，單句）：

```
音長 2.68 s → 合成 3.61 s
音長 3.84 s → 合成 8.13 s
音長 4.56 s → 合成 5.28 s
```

第一次請求要多花約 20 秒載入模型。併發 4 個會依序完成（3.4 / 6.7 / 10.1 / 13.0 秒），
全部 200。**CastAgent 的 `_TTS_CONCURRENCY=4` 可以維持**——不會失敗，只是排隊，
而排隊比並行搶顯存快。

## 長文

模型只在 30 秒以內的樣本上訓練過（所有語料的 `max-duration` 都是 30 秒），整段丟
進去會生出講不完的音檔：實測 211 字不分段只出 41 秒，聽起來不知道在講什麼。

所以超過一句就自動分段（用 CosyVoice 的 `split_paragraph`，按標點切、每段約 80
字），逐段合成再串接。538 字切成 7 段、音長 127 秒、耗時 139 秒。

**預設只在句末標點切**（`。？！；：`），逗號不切。整段只用逗號串連的文字——新聞稿
常這樣寫——會維持單段送進模型，超過 80 字就可能講不完。

`COMMA_SPLIT=1` 可以改成在逗號也切，但**預設關閉**，因為兩種失敗都難聽而後者更難
察覺：在逗號切會把一句話從中間剖開，前半被當成句尾唸完收尾、後半又從頭起一句，
語意還在延續而語調已經結束又重來，接縫處聽得出明顯的跳。**漏字聽得出來，跳掉的
語調聽起來像是模型本來就這樣講**，所以留給人決定，不預設打開。

呼叫端最可靠的做法是自己先斷句：在語意完整的地方放句號，分段點就是你決定的。

分段只借切段函式，**不啟用 CosyVoice 的其他正規化**——那會做數字轉寫、去括號、
刪除漢字與拉丁字母之間的空格，對台羅標音有影響，也讓實驗對照多出變數。

排隊上限 600 秒。實測一個長文加三個短句同時送，最後一個 218 秒完成，全部 200。

## 設定

| 環境變數 | 預設 | 說明 |
|---|---|---|
| `TTS_API_KEY` | 無 | **必設**，否則 `/v1` 停用 |
| `TTS_API_LLM` | `llm_v7_best` | 掛哪個 LLM checkpoint |
| `TTS_API_FLOW` / `TTS_API_HIFT` | `official` | 這兩個微調過沒有比較好 |
| `TTS_API_SIMPLIFY` | `full` | 繁→簡字形轉換，只換字形不換用詞 |
| `TTS_API_MAX_WAIT_SEC` | `600` | 排隊超過就回 429。呼叫端的 timeout 要比它大 |
| `TTS_API_MAX_TEXT_CHARS` | `5000` | 文字上限 |
| `TTS_API_CHUNK_CHARS` | `80` | 每段字數，長文照這個切 |
| `COMMA_SPLIT` | `0` | 是否在逗號也切段（見上）。試聽台與 `/v1` 共用這個設定 |

換模型改環境變數重啟即可，`model_version` 會跟著變。

## 沒有實作的

**`POST /v1/synthesize/batch`** —— 規格列為建議。它的主要好處是「本地模型可避免
重複載入」，但這裡模型常駐，省下的只有 HTTP 來回；而回傳 N 個音檔需要 multipart
或 zip，對呼叫端反而更麻煩。之後真的變成瓶頸再加。

**`pitch`** —— CosyVoice3 的 `inference_instruct2` 沒有這個參數。語速有，音高沒有。
