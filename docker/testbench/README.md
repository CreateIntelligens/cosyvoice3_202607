# CosyVoice3 台語模型試聽台

把微調出來的 llm / flow / hift checkpoint 任意搭配，和官方原始模型並排試聽。

---

## 快速開始（五個步驟）

### 1. 前置需求

```bash
docker --version                      # 需要 Docker + compose
nvidia-smi                            # 驅動要正常，NVML 版本不匹配就得先重開機
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
docker images | grep cosyvoice3-taipei1   # 需要 happy74527/cosyvoice3-taipei1:latest
```

image 若不在本機：`docker pull happy74527/cosyvoice3-taipei1:latest`（21 GB）。

### 2. 建立模型目錄

在 repo 根目錄執行：

```bash
cd /home/hank/Cosyvoice3
mkdir -p models/base models/variants/{llm,flow,hift} models/prompts
```

`models/base` 要是**空目錄**，不要放東西也不要做 symlink——它只是掛載點，compose
會把真正的官方模型掛上去（symlink 在容器內會指向不存在的路徑）。

最終結構：

```text
models/
├── base/            空目錄，compose 會掛上官方 Fun-CosyVoice3-0.5B
├── variants/
│   ├── llm/         llm_best.pt, llm_avg5.pt, ...
│   ├── flow/        flow_best.pt, flow_avg5.pt, ...
│   └── hift/        HiFT 練好前留空，選單只會有 official
└── prompts/         參考音色 *.wav（決定音色，不是要唸的內容）
```

下拉選單直接讀 `variants/<part>/*.pt` 的**檔名**，丟進去就會出現，不用改程式也不用
重啟容器（重新整理網頁即可）。每個選單一定有 `official`，指向官方權重，那就是對照組。

### 3. 放官方模型

compose 預設從 `../../workspace/cosyvoice3_smoke_10spk/models/Fun-CosyVoice3-0.5B`
掛載。若你的官方模型在別處：

```bash
export BASE_MODEL_DIR=/絕對路徑/Fun-CosyVoice3-0.5B
```

這個目錄至少要有 `llm.pt`、`flow.pt`、`hift.pt`、`cosyvoice3.yaml`、`campplus.onnx`、
`speech_tokenizer_v3.onnx`、`CosyVoice-BlankEN/`。（`spk2info.pt` 沒有也沒關係。）

### 4. 取回微調 checkpoint 與參考音色

```bash
SSH_OPT="-e \"ssh -i ~/.ssh/Hank_ed25519 -p 2222 -o 'ProxyCommand=ssh -i ~/.ssh/Hank_ed25519 -W %h:%p happy74527-aicr-48624a@jb1.frontier.nvidia.com -p 2222'\""
TP1=happy74527-aicr-48624a@1.tun.tp1-cluster.nvidia.com

# checkpoint（約 7.8 GB，建議放 tmux）
rsync -a --progress --partial --append-verify $SSH_OPT \
  $TP1:/mnt/shared/p06/cosyvoice/taigi_v1/export/ /tmp/tp1_export/
cp /tmp/tp1_export/llm_*.pt  models/variants/llm/
cp /tmp/tp1_export/flow_*.pt models/variants/flow/

# 參考音色
rsync -a $SSH_OPT $TP1:/mnt/shared/p06/cosyvoice/taigi_v1/prompts/ models/prompts/
```

`*_best.pt` 是 CV loss 最低的單一 epoch，`*_avg5.pt` 是 CV 最佳五個 epoch 的平均。

參考音色也可以自己準備：3–10 秒、單人、乾淨的 `.wav` 放進 `models/prompts/`，或直接在
網頁上傳。**硬上限 30 秒**（speech tokenizer 的限制）。

### 5. 啟動

```bash
cd docker/testbench
docker compose up -d
docker compose logs -f            # 等到 "Uvicorn running on http://0.0.0.0:50010"
```

開 <http://localhost:50010>。頁面頂端會顯示 GPU 型號與各部位可選的 checkpoint 數量；
若出現「⚠️ 找不到官方模型」代表第 3 步的掛載路徑不對。

停止／重啟：

```bash
docker compose down
docker compose restart            # 改過 app/ 底下的檔案後用這個（程式碼是掛載進去的）
```

---

## 網頁怎麼用

1. **選 checkpoint**：llm / flow / hift 三個下拉各自獨立，可任意組合
2. **輸入華語文字**：模型會輸出台語語音
3. **選音色**：預設清單或自行上傳；兩者都能**先試聽**確認來源乾不乾淨
4. **固定 seed**（重要）：比較 checkpoint 時務必固定，否則差異裡混雜了隨機取樣
5. 按「合成」，結果會堆疊在右側，可反覆比較

音色來源有三層視覺提示避免搞混：上傳後按鈕變綠、狀態框變綠、下拉選單變灰停用。
按「取消上傳」即可回到預設清單。

建議的對照組合（同一個 seed、同一段文字、同一個音色）：

| llm | flow | hift | 回答的問題 |
|---|---|---|---|
| official | official | official | 基準線，原始模型對台語的表現 |
| `llm_best` | official | official | LLM 微調有沒有用 |
| official | `flow_best` | official | flow 微調有沒有用 |
| `llm_best` | `flow_best` | official | 兩者疊加 |
| `llm_avg5` | `flow_avg5` | official | 平均版是否比單一最佳更穩 |

音色要選 **holdout** 開頭的那幾個（訓練完全沒看過），那才是真正的 zero-shot 條件。

---

## API

```bash
curl -X POST http://localhost:50010/api/tts \
  -F text="明天早上九點，我們一起去市場買東西。" \
  -F llm=llm_best -F flow=flow_best -F hift=official \
  -F prompt_name=holdout_male_drama1_680.wav \
  -F seed=1234 \
  -o out.wav
```

| 端點 | 用途 |
|---|---|
| `GET /api/variants` | 可選的 checkpoint、音色、預設 instruction、長度限制 |
| `GET /api/health` | CUDA 是否可用、目前載入的組合 |
| `GET /api/prompt/<name>` | 取得預設音色檔（網頁試聽用） |
| `POST /api/tts` | 合成；`prompt_wav` 上傳檔或 `prompt_name` 選預設 |

`POST /api/tts` 參數：`text`、`llm`、`flow`、`hift`、`instruct`、`prompt_name` 或
`prompt_wav`、`speed`、`text_frontend`、`seed`。

### Seed

`seed` 預設 `-1`（每次隨機），合法範圍 `0 ~ 2147483647`。給定同一組 checkpoint、文字、
音色與 seed，輸出可**完全重現**（實測兩次 md5 相同）。推論的隨機只有 LLM 取樣的
`multinomial` 與 flow matching 的 `torch.randn_like`，兩者都走 torch 全域 RNG。

回應的 `X-Seed` header 會回報實際用到的值，即使你沒指定——所以隨機聽到滿意的結果，
事後一樣追得回來。後端日誌也會記錄完整條件：

```bash
docker logs cosyvoice3-testbench 2>&1 | grep "tts seed="
# INFO tts seed=827307999 llm=llm_best flow=flow_best hift=official
#      prompt=xxx.wav speed=1.0 text='明天早上九點，我們一起去市場買東西。'
```

注意 seed 只保證**同一組 checkpoint** 可重現；換 checkpoint 後機率分布不同，取樣路徑
本來就會分岔，這不是 bug。

---

## 疑難排解

| 症狀 | 原因與解法 |
|---|---|
| 頁面顯示「⚠️ 找不到官方模型」 | `models/base` 沒掛到東西。檢查 `BASE_MODEL_DIR`，且主機端 `models/base` 必須先存在（`/models` 是唯讀掛載，Docker 無法自己建子目錄） |
| 選單只有 `official` | `models/variants/<part>/` 是空的，或檔案不是 `.pt`。放進去後重新整理網頁即可 |
| 頁面顯示「⚠️ 沒有 CUDA」 | 容器拿不到 GPU。先在主機確認 `nvidia-smi`，再確認 `docker run --gpus all` 能動 |
| `all predefined address pools have been fully subnetted` | Docker 網段用盡。compose 已用 `network_mode: host` 迴避，不要改回 `ports:` |
| 上傳音檔後回報「超過 30 秒」 | `frontend.py` 的硬限制。用 `ffmpeg -i in.mp3 -t 8 -ar 24000 -ac 1 out.wav` 裁切 |
| 合成很慢（20 秒以上） | 切換 checkpoint 需要重新載入模型。同一組合連續合成約 6–7 秒 |
| 改了 `app/` 的程式沒生效 | `docker compose restart`（程式碼是唯讀掛載，不需重建 image） |

---

## 注意事項

- **一次只常駐一個組合。** 三個模組加起來要好幾 GB VRAM，而這台的 GPU 還跑著
  `cosyvoice-service` 等服務（20 GB 卡上僅約 9 GB 可用），所以換組合時會先釋放前一個
  再載入。這是刻意的取捨，不是效能問題。
- **instruction 必須和訓練時完全一致**，介面預設已帶上
  `You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>`。
  改掉它等於換了一個任務條件，輸出會退化。
- **文字正規化預設關閉**（`text_frontend=false`）。既有的中文 normalizer 會擅自改寫
  台語用字，見 `docs/taigi-universal-tts-training.md` §13。
- 容器把工作目錄掛成唯讀的 `/opt/CosyVoice`，所以 repo 裡的修正（flow 長度對齊、
  GAN dataloader、average_model 去重）會直接生效，**不需要重建 21 GB 的 image**。
