# CosyVoice3 台語模型試聽台

把微調出來的 llm / flow / hift checkpoint 任意搭配，和官方原始模型並排試聽。

## 目錄結構

`MODELS_DIR`（預設 `<repo>/models`）底下要長這樣：

```text
models/
├── base/                       官方 Fun-CosyVoice3-0.5B，整份
│   ├── llm.pt  flow.pt  hift.pt
│   ├── cosyvoice3.yaml  campplus.onnx  speech_tokenizer_v3.onnx
│   └── CosyVoice-BlankEN/
├── variants/
│   ├── llm/                    llm_best.pt, llm_avg5.pt, ...
│   ├── flow/                   flow_best.pt, flow_avg5.pt, ...
│   └── hift/                   （HiFT 練好之前留空，介面只會有 official）
└── prompts/                    參考音色 *.wav（決定音色，不是要唸的內容）
```

下拉選單直接讀 `variants/<part>/*.pt` 的檔名，**丟進去就會出現，不用改程式**。每個
選單一定有 `official`，指向 `base/` 裡的原始權重，這就是對照組。

## 準備模型

官方模型（本機已有一份）：

```bash
mkdir -p models
ln -s /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/models/Fun-CosyVoice3-0.5B models/base
```

從 Taipei-1 取回微調 checkpoint（已用 CV loss 挑好並平均）：

```bash
mkdir -p models/variants/{llm,flow,hift} models/prompts
rsync -av --progress \
  -e "ssh -i ~/.ssh/Hank_ed25519 -p 2222 -o 'ProxyCommand=ssh -i ~/.ssh/Hank_ed25519 -W %h:%p happy74527-aicr-48624a@jb1.frontier.nvidia.com -p 2222'" \
  happy74527-aicr-48624a@1.tun.tp1-cluster.nvidia.com:/mnt/shared/p06/cosyvoice/taigi_v1/export/ /tmp/export/

mv /tmp/export/llm_*.pt  models/variants/llm/
mv /tmp/export/flow_*.pt models/variants/flow/
```

`*_best.pt` 是 CV loss 最低的單一 epoch，`*_avg5.pt` 是 CV 最佳五個 epoch 的平均。

參考音色放幾個 `.wav` 到 `models/prompts/`（3–10 秒、單人、乾淨即可）。也可以在
網頁上直接上傳。

## 啟動

```bash
cd docker/testbench
docker compose up -d
docker compose logs -f          # 等到 Uvicorn running
```

開 <http://localhost:50010>。

## API

```bash
curl -X POST http://localhost:50010/api/tts \
  -F text="明天早上九點，我們一起去市場買東西。" \
  -F llm=llm_best -F flow=flow_best -F hift=official \
  -F prompt_name=speaker_a.wav \
  -o out.wav
```

| 端點 | 用途 |
|---|---|
| `GET /api/variants` | 目前可選的 checkpoint、音色、預設 instruction |
| `GET /api/health` | CUDA 是否可用、目前載入的組合 |
| `POST /api/tts` | 合成；`prompt_wav` 上傳檔或 `prompt_name` 選預設 |

`POST /api/tts` 的 `seed` 預設 `-1`（每次隨機）。給定同一組 checkpoint、文字、音色與
seed，輸出可完全重現（實測兩次 md5 相同）。推論的隨機只有 LLM 取樣的 `multinomial`
與 flow matching 的 `torch.randn_like`，兩者都走 torch 全域 RNG。回應的 `X-Seed`
header 會回報實際用到的值，即使沒指定。

比較 checkpoint 時**務必固定 seed**，否則聽到的差異裡混雜了隨機取樣，分不出是模型
造成的。文件 §14.2 也要求固定 prompt、文字與 seed。注意 seed 只保證同一組 checkpoint
可重現；換 checkpoint 後機率分布不同，取樣路徑本來就會分岔。

## 注意事項

- **一次只常駐一個組合。** 三個模組加起來要好幾 GB VRAM，而這台的 GPU 還跑著
  `cosyvoice-service` 等服務，所以換組合時會先釋放前一個再載入，第一次合成會比較久。
- **instruction 必須和訓練時完全一致**，介面預設已經帶上
  `You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>`。
  改掉它等於換了一個任務條件，輸出會退化。
- **文字正規化預設關閉**（`text_frontend=false`）。既有的中文 normalizer 會擅自改寫
  台語用字，見 docs/taigi-universal-tts-training.md §13。
- 容器把工作目錄掛成唯讀的 `/opt/CosyVoice`，所以 repo 裡的修正（flow 長度對齊、
  GAN dataloader、average_model 去重）會直接生效，不需要重建 21 GB 的 image。
