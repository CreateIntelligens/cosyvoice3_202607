# CosyVoice3 通用臺灣台語 TTS：資料、容器與 Taipei-1 訓練手冊

## 0. 目標與結論

本計畫的使用介面只接收華語中文，輸出臺灣台語語音。現有原始資料為：

- 三立八點檔《願望》168 集，去除空白後約 168 小時。
- 《鳥來伯與十三姨》71 集，去除空白後約 17.75 小時。
- 合計約 185.75 小時、多說話者、戲劇對話資料。
- 音軌是臺灣台語、訓練文字是已對齊的華語字幕。
- 185.75 小時已完成字幕錯配、多人重疊、配樂及分離偽影排除，是可用淨資料時數，不是清洗前 gross hours。

核心判斷：

1. 正式模型路徑固定為「華語字幕直接 → 臺灣台語語音」，推論時不產生台文或台羅中間結果。
2. 185.75 小時乾淨、多說話者資料足以進行完整 LLM 與 flow 適配；第一版仍應先用小 subset 驗證資料格式與 loss。
3. CosyVoice3 LLM 會直接學習華語文字 token 到臺灣台語 speech token 的條件映射，也就是把語意轉換與語音生成一起學進模型。
4. 這批資料的主要限制不再是清洗量，而是戲劇語域偏差。要達到跨場景「通用」，後續仍應補中性敘述、新聞、數字地址、生活用語、不同年齡與腔口。
5. 第一版只微調 `llm`，第二版視聲學問題微調 `flow`；預設保留官方 `hift.pt`，不要把 HiFi-GAN 當成必要步驟。

本手冊以官方 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` base checkpoint 為起點。

---

## 1. 目標系統架構

```text
華語字幕＋「翻譯成臺灣台語並朗讀」指令
  │
  ▼
CosyVoice3 LLM
  │
  ▼
臺灣台語 speech tokens → flow → HiFT
```

這在模型結構上可行：CosyVoice3 LLM 的 loss 是讓文字條件預測 speech tokens，並不要求輸入文字與發音逐字同語言。模型不輸出台文或台羅，內部直接自回歸生成臺灣台語語音 token。

必須接受的模型特性：

- 0.5B TTS LLM 不是專用翻譯器。
- 即使資料已清理，華語字幕若是意譯，模型學的是語意到語音而非逐字 grapheme-to-phoneme；這正是目標，但評估必須包含語意正確性。
- 185.75 小時集中在兩部戲劇，未見華語句型與非戲劇詞彙的泛化仍需實測。

訓練 manifest 的 `text` 只放華語字幕。metadata 可保留原始字幕與正規化字幕兩欄以便追溯，但不要求建立台文或台羅。

---

## 2. 權利與資料治理是訓練前置閘門

電視劇影像、音訊、字幕、演員聲音及衍生模型可能分別涉及著作權、契約、個資或聲音人格權。即使檔案可取得，也不等於可以上傳到 Taipei-1、訓練、公開 checkpoint 或商用。

在開始前取得並保存下列文件：

- 音訊及字幕的訓練使用授權。
- 是否允許第三方運算平台處理資料。
- 是否允許生成式模型、聲音複製及衍生 checkpoint。
- 模型能否公開、商用或只限內部研究。
- 資料與 checkpoint 的保存期限及刪除規則。
- 演員／配音員聲音相關權利的處理依據。

若授權尚未確認，只能先用自有或明確授權的小資料驗證 pipeline。不要把未確認授權的影視資料、模型或中間 speech tokens 推到公開 registry。

---

## 3. 已清理資料的最終驗收項目

185.75 小時已完成主要清理，以下不是要求重新清理，而是建立 manifest 前的驗收清單，避免時間軸或處理版本混入。

### 3.1 去除空白音軌與時間軸

若「去除空白」是把整段節目中的靜音直接刪掉並重新串接，原字幕 timestamp 將不再對應音訊。正確作法是：

1. 以原始、未改變時間軸的音訊和字幕做切段。
2. 每個切好的 utterance 內再修剪頭尾靜音。
3. 若只能保留去空白長音訊，必須同時有 edit decision list（EDL）或原時間到新時間的映射。
4. 若 EDL 和原音訊都不存在，只能重新做 VAD、ASR、語意匹配與人工校正，成本會顯著提高。

### 3.2 背景聲移除可能傷害 vocoder 訓練

source separation／降噪常留下：

- 金屬音、musical noise。
- 擦音與高頻被削掉。
- 音量抽動。
- 角色聲音被部分移除。
- 配樂殘留或相位偽影。

LLM 對這些問題相對不敏感，flow 和 HiFi-GAN 會直接學到它們。必須保留原始音訊，並在 manifest 紀錄 `cleaning_method`、模型版本及品質分數。

### 3.3 字幕不是逐字稿

常見錯配包括：

- 台語說十幾個音節，字幕只有短短一句華語摘要。
- 一張字幕對應兩個角色。
- 台語夾華語，但字幕全部正規化成華語。
- 人名、語助詞、髒話或重複詞被刪除。
- 字幕跨越鏡頭或提前出現。

直接餵入會導致漏字、重複、錯誤停頓及文字條件失效。

---

## 4. 建立可訓練 utterance

### 4.1 建議處理順序

```text
原始節目音訊＋字幕
  → 字幕時間區間候選
  → VAD 修正語音邊界
  → speaker diarization
  → 重疊語音／配樂／音效偵測
  → 臺灣台語 ASR 草稿
  → 字幕與 ASR 的跨語言語意比對
  → 分層人工抽驗 A/B 資料
  → 音量、時長、取樣率與文字 QA
  → train/dev/test
```

### 4.2 切段規格

- Mono。
- 原始取樣率至少 16 kHz；CosyVoice3 pipeline 會轉為 24 kHz。
- 建議 2–15 秒，硬上限 30 秒。
- 頭尾各保留約 100–250 ms 自然空白，不要硬切爆破音。
- 單一主要說話者。
- 重疊說話、明顯背景音樂、笑聲蓋住語句、電話濾波和嚴重混響分開標記。
- 不把不同角色串成同一 utterance。
- peak 不 clipping，並避免逐句做不一致的 loudness normalization。

專案的 `extract_speech_token.py` 對超過 30 秒的音訊會輸出空 token，所以 30 秒是實際硬限制。

### 4.3 Speaker ID

以「節目＋角色／演員＋錄音域」建立 speaker ID，例如：

```text
wish_actor_001
wish_actor_002
bird_actor_001
```

同一演員在不同節目或差異極大的錄音域，第一版先分成不同 ID。確認 clustering 後再合併。不要只用字幕角色名而不抽查聲音，因為畫外音和字幕標記可能錯誤。

### 4.4 品質分級

每個 utterance 建議保存以下 metadata：

```text
utt_id, show, episode, start, end, speaker_id,
text_zh_raw, text_zh_normalized,
language_mix, overlap, music, sfx, denoise_artifact,
alignment_score, qc_status, split, license_scope
```

分級：

- A：單人、乾淨、華語字幕與臺灣台語語音語意完整對齊。
- B：已通過清理，但有較強情緒、口音、通道或輕微語意壓縮。
- Reject：任何重新檢查時發現的多人重疊、錯字幕、音樂／分離偽影或無法辨識片段。

用途：

| 模組 | 可用資料 |
|---|---|
| 華語直接到台語 LLM | 全部 185.75 小時 A/B 資料 |
| Flow | 全部聲學驗收合格資料；文字不直接進入 flow |
| HiFi-GAN | 優先只用聲學最乾淨的 A 級資料 |

185.75 小時就是本計畫的 retained clean hours。仍需保存 A/B 分布與每部戲、每位 speaker 的時數，讓實驗可重現。

---

## 5. 華語字幕直接到臺灣台語語音的文字策略

CosyVoice manifest 的 `text` 一律使用華語字幕，不建立台文或台羅中間資料：

```text
明天早上九點，我們一起去市場買東西。
```

`instruct` 固定使用同一個任務描述：

```text
You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>
```

訓練和推論必須使用完全相同的 instruction，`<|endofprompt|>` 不可省略。第一版不要混用「請用閩南話」「請用台語朗讀」等不同提示，避免模型把相同任務學成多個條件。

華語字幕正規化只處理輸入形式，不做台語翻譯：

- 統一全形／半形、標點與不可見字元。
- 明確規定阿拉伯數字、日期、時間、百分比和貨幣的華語輸入形式。
- 保留人名、地名與專有名詞，不自動換成台語文字。
- 同時保存 `text_zh_raw` 和 `text_zh_normalized`。
- 訓練與推論使用同一個 normalizer。

CosyVoice3 將華語 text tokens 與 instruction tokens 編碼後，直接預測對應的臺灣台語 speech tokens；flow 和 HiFT 完全不需要看到台文或台羅。

---

## 6. Train/dev/test 設計

不可逐句隨機切分，否則相鄰鏡頭、同句台詞、背景與角色會洩漏。

至少保留以下測試集：

1. `test_seen_speaker`：見過的角色、未見過的完整集數。
2. `test_unseen_speaker`：訓練完全未出現的角色，用於 zero-shot。
3. `test_unseen_episode`：以整集為單位 hold out。
4. `test_neutral_text`：非戲劇語域的新聞、生活、數字、地址和說明文。
5. `test_zh2taigi`：華語輸入，另有人工臺灣台語參考翻譯和錄音。

建議先保留：

- Train：依完整 episode 分配約 85–90%。
- Dev：完整 episode 約 5%。
- Test：完整 episode 約 5–10%。
- 另保留至少數名完全 unseen speakers。

兩部戲都應出現在 train/dev/test，但不能讓同一 episode 橫跨 split。另建立跨劇測試，檢查模型是否只記住單一節目的聲學域。

---

## 7. CosyVoice manifest

每個 split 建立：

```text
data/taigi_train/
├── wav.scp
├── text
├── utt2spk
├── spk2utt
└── instruct
```

`wav.scp`：

```text
wish_e001_000001 /datasets/taigi/wav/wish_e001_000001.wav
```

`text` 直接填華語字幕：

```text
wish_e001_000001 今天天氣很好，我們一起去逛夜市。
```

`utt2spk`：

```text
wish_e001_000001 wish_actor_001
```

`spk2utt`：

```text
wish_actor_001 wish_e001_000001 wish_e001_000008
```

`instruct`：

```text
wish_e001_000001 You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>
```

只建立華語字幕直接到臺灣台語語音這一條訓練線；原始與正規化華語字幕仍分欄保存，Parquet 的 `text` 使用正規化華語。

---

## 8. Taipei-1 容器策略

Taipei-1 是 DGX H100 SuperPOD；實際 image registry、Slurm/Pyxis/Enroot 參數、掛載路徑及網路政策應以核發帳號時提供的專案文件為準。不要假設 compute node 可以連網或執行 `conda install`。

原則：

- image 內包含 OS、Python、CUDA userspace、PyTorch、ONNX Runtime、ffmpeg、sox、pyworld、Matcha-TTS 和 CosyVoice 程式。
- 模型、資料、Parquet、checkpoint 和 log 放共享高速儲存，不烘進 image。
- image 內不依賴互動式 `conda activate`。
- 固定 repository commit 與 Python package 版本。
- 在送到 Taipei-1 前，於另一台 NVIDIA GPU 主機完成 smoke test。
- 若平台使用 Enroot/Pyxis，通常從 registry image 匯入；若要求 `.sqsh` 或其他格式，再依平台文件轉換。

### 8.1 建議 Dockerfile

在 repository root 建置；此範例不使用 Conda：

```dockerfile
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    git git-lfs build-essential cmake ninja-build pkg-config \
    ffmpeg sox libsox-dev libsndfile1-dev \
    curl wget unzip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --upgrade pip setuptools wheel

WORKDIR /opt/CosyVoice
COPY . /opt/CosyVoice

RUN test -f third_party/Matcha-TTS/matcha/__init__.py
RUN python3.10 -m pip install -r requirements.txt

ENV PYTHONPATH=/opt/CosyVoice:/opt/CosyVoice/third_party/Matcha-TTS
WORKDIR /workspace

CMD ["/bin/bash"]
```

建置前必須初始化 submodule：

```bash
git submodule update --init --recursive
docker build -f docker/Dockerfile.taigi-train -t REGISTRY/cosyvoice3-taigi:COMMIT .
```

不要用 `latest` 作正式實驗 tag。記錄 image digest：

```bash
docker inspect --format='{{index .RepoDigests 0}}' REGISTRY/cosyvoice3-taigi:COMMIT
```

### 8.2 Image smoke test

CPU/import 測試：

```bash
docker run --rm REGISTRY/cosyvoice3-taigi:COMMIT \
  python3.10 -c "import torch, torchaudio, onnxruntime, pyworld, matcha, cosyvoice; print(torch.__version__, onnxruntime.get_available_providers())"
```

GPU 測試：

```bash
docker run --rm --gpus all REGISTRY/cosyvoice3-taigi:COMMIT \
  python3.10 -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())"
```

功能測試必須在上傳前完成：

- 載入官方 CosyVoice3 並合成一小句。
- `extract_embedding.py` 處理一個 WAV。
- `extract_speech_token.py` 處理一個 WAV，確認使用 `CUDAExecutionProvider`。
- 用 10–100 筆 tiny dataset 各跑一次 LLM、flow、HiFi-GAN forward/backward。
- checkpoint 寫入掛載目錄，停止後可再次載入。

### 8.3 掛載規劃

容器內固定使用：

```text
/workspace/project       程式與設定
/workspace/models        官方模型與輸出 checkpoint
/workspace/data          Parquet 或切段後語音
/workspace/exp           checkpoint、TensorBoard、log
```

Taipei-1 實際啟動命令可能是 Slurm + Pyxis/Enroot，也可能由平台 wrapper 提交。以下只是結構示意，不可在未知平台參數下原樣提交：

```bash
srun <Taipei-1 allocation options> \
  --container-image=REGISTRY/cosyvoice3-taigi:COMMIT \
  --container-mounts=HOST_PROJECT:/workspace/project,HOST_DATA:/workspace/data,HOST_MODELS:/workspace/models,HOST_EXP:/workspace/exp \
  bash /workspace/project/jobs/train_llm.sh
```

需要向 Taipei-1 管理者確認：registry 登入、image size 上限、共享檔案系統、節點網路、每節點 8 張 H100 的可見方式、NCCL/InfiniBand 設定、作業最長時間、checkpoint 保留與 job requeue。

---

## 9. 模型與資料前處理

下載官方 base model 到共享儲存：

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    local_dir="/workspace/models/Fun-CosyVoice3-0.5B"
)
```

若 compute node 不可連網，在可連網節點先下載、檢查完整性，再傳入共享儲存。

對 train/dev/test 分別執行：

```bash
python /opt/CosyVoice/tools/extract_embedding.py \
  --dir /workspace/data/taigi_train \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B/campplus.onnx \
  --num_thread 16

python /opt/CosyVoice/tools/extract_speech_token.py \
  --dir /workspace/data/taigi_train \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B/speech_tokenizer_v3.onnx \
  --num_thread 16

mkdir -p /workspace/data/taigi_train/parquet
python /opt/CosyVoice/tools/make_parquet_list.py \
  --num_utts_per_parquet 1000 \
  --num_processes 16 \
  --src_dir /workspace/data/taigi_train \
  --des_dir /workspace/data/taigi_train/parquet
```

預先抽 token 能降低多 GPU training 時重複執行 ONNX 的負擔。完成後抽查 Parquet 欄位、音訊、文字、speaker embedding 和 speech token 非空。

---

## 10. 設定檔

從 `examples/libritts/cosyvoice3/conf/cosyvoice3.yaml` 複製實驗設定，不直接修改官方檔案。

### 10.1 LLM/flow 起始設定

```yaml
padding:
    use_spk_embedding: False

batch:
    batch_type: dynamic
    max_frames_in_batch: 2000

train_conf:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    max_epoch: 20
    grad_clip: 5
    accum_grad: 2
    log_interval: 50
    save_per_step: 1000
```

這是多說話者、保留 zero-shot 能力的起點，所以維持 `use_spk_embedding: False`，使用每段 utterance embedding。若未來做固定單一聲音 SFT，才另行比較 `True`。

`max_frames_in_batch` 是每個 rank 的動態 batch 預算；先用單 GPU tiny run 測試，再依 H100 數量及顯存調整。多 GPU 不應盲目把每卡 batch 同時放大，避免 global batch 暴增。

---

## 11. 訓練階段

### 11.1 Stage A：華語字幕直接到臺灣台語 LLM

目的：從官方 CosyVoice3 `llm.pt` 開始，直接學習華語字幕到臺灣台語 speech token 的映射。不需要台語逐字稿預訓練階段。

```bash
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  /opt/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /workspace/project/conf/taigi_cosyvoice3.yaml \
  --train_data /workspace/data/taigi_zh2taigi_train.data.list \
  --cv_data /workspace/data/taigi_zh2taigi_dev.data.list \
  --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
  --model llm \
  --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir /workspace/exp/taigi_zh2taigi/llm \
  --tensorboard_dir /workspace/exp/taigi_zh2taigi/tensorboard/llm \
  --ddp.dist_backend nccl \
  --num_workers 8 --prefetch 20 --pin_memory --use_amp
```

H100 可使用 BF16。先確認一個 epoch 的 train/CV loss、梯度和固定句合成正常，再擴大到多節點。


### 11.2 Stage B：Flow（選配但比 HiFi-GAN優先）

Flow 學習 speech token、speaker embedding 到 mel。若 LLM 輸出的內容正確，但聲調、韻律或臺灣台語音色仍明顯不自然，再訓練 flow：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  /opt/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /workspace/project/conf/taigi_cosyvoice3.yaml \
  --train_data /workspace/data/taigi_audio_train.data.list \
  --cv_data /workspace/data/taigi_audio_dev.data.list \
  --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
  --model flow \
  --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir /workspace/exp/taigi/flow \
  --tensorboard_dir /workspace/exp/taigi/tensorboard/flow \
  --ddp.dist_backend nccl \
  --num_workers 8 --prefetch 20 --pin_memory --use_amp
```

Flow 使用這 185.75 小時已通過聲學驗收的資料。華語字幕不會直接進入 flow loss；flow 只看 speech token、mel 與 speaker embedding。

### 11.3 Checkpoint 平均

```bash
python /opt/CosyVoice/cosyvoice/bin/average_model.py \
  --dst_model /workspace/exp/taigi_zh2taigi/llm.pt \
  --src_path /workspace/exp/taigi_zh2taigi/llm \
  --num 5 --val_best
```

官方 `examples/libritts/cosyvoice3/run.sh` 的 averaging 路徑仍寫成 `exp/cosyvoice`，和訓練輸出的 `exp/cosyvoice3` 不一致，不應原樣執行。

---

## 12. HiFi-GAN／HiFT：能不能訓練？

### 12.1 答案

CosyVoice3 程式確實包含可訓練的 `CausalHiFTGenerator`、多週期／多解析度 discriminator、mel reconstruction、feature matching、F0 和 TPR loss，因此「程式上可以訓練」。但官方 recipe 的 HiFi-GAN 路徑不夠完整，不能把 `run.sh` 原樣執行後期待安全的 SFT。

對本計畫而言，第一版應保留官方 `hift.pt`，原因是：

- Vocoder 大致語言無關；臺灣台語適配的主要收益通常先來自 LLM 和 flow。
- 官方 vocoder 已見過大規模、多語言資料，遠多於這批經過降噪的戲劇音訊。
- 戲劇音訊及 source separation 偽影很容易被 GAN 放大。
- GAN fine-tuning 比 LLM/flow 更容易因 learning rate、AMP、F0、裁切及 checkpoint 載入錯誤而崩壞。

### 12.2 推論與訓練使用的是兩種 state-dict 結構

先直接回答：

- 官方模型做推論：不需要轉換，`CosyVoice3` 直接讀裸 generator 的 `hift.pt`。
- 官方模型拿來繼續訓練 HiFi-GAN：需要建立 wrapper checkpoint，但不是把 `hift.pt` 單純改名成 `hifigan.pt`；必須把每個 state-dict key 加上 `generator.`。輸出檔名其實可以任意取，這份手冊叫它 `hifigan_init.pt` 只是避免混淆。
- 訓練完成部署：反向抽出 `generator.*` 並移除 prefix，存回推論用 `hift.pt`。

官方 `run.sh` 對每個 model 使用：

```text
--checkpoint $pretrained_model_dir/$model.pt
```

當 `$model` 是 `hifigan` 時，它尋找 `hifigan.pt`；官方推論模型通常提供的是 `hift.pt`。找不到時，`train.py` 只記 warning，訓練仍繼續，GAN generator 可能從隨機初始化開始。

此外，訓練物件是 wrapper：

```text
generator.*
discriminator.*
```

而官方 `hift.pt` 是裸 generator state dict。直接傳入時，因 `strict=False`，程式不會報錯，但 key 名稱不匹配也可能導致 generator 幾乎沒有載入。若你當時已經正確加入 `generator.` prefix，並驗證所有 generator keys 都成功載入，那麼這就不是當次損壞原因；下一順位應檢查 GAN learning rate、是否使用 AMP、F0 extraction、mel/sample-rate 參數、裁切長度，以及訓練後是否把 discriminator keys 一起塞進推論 `hift.pt`。

### 12.3 正確建立 GAN 初始 checkpoint

從官方 `hift.pt` 加上 `generator.` prefix：

```python
import torch

src = torch.load("/workspace/models/Fun-CosyVoice3-0.5B/hift.pt", map_location="cpu")
dst = {f"generator.{key}": value for key, value in src.items()}
torch.save(dst, "/workspace/models/Fun-CosyVoice3-0.5B/hifigan_init.pt")
```

必須增加載入驗證：比較 `hifigan_init.pt` 的 `generator.*` keys 與 `configs['hifigan'].generator.state_dict()`，確認 generator 參數 100% 覆蓋；discriminator 隨機初始化是預期行為。不要只看 `load_state_dict(strict=False)` 沒有 exception。

### 12.4 若要做實驗，使用保守設定

先用 A 級音訊做小規模試驗；A/B 全量是否加入由 reconstruction 測試決定：

```yaml
truncate:
    truncate_length: 48000  # 2 秒，且可被 960 整除

train_conf_gan:
    optim: adam
    optim_conf:
        lr: 1e-5
    scheduler: constantlr
    optim_d: adam
    optim_conf_d:
        lr: 1e-4
    scheduler_d: constantlr
    max_epoch: 10
    grad_clip: 5
    accum_grad: 1
    log_interval: 20
    save_per_step: 1000
```

啟動時使用：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
  /opt/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --config /workspace/project/conf/taigi_cosyvoice3.yaml \
  --train_data /workspace/data/taigi_hifigan_train.data.list \
  --cv_data /workspace/data/taigi_hifigan_dev.data.list \
  --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
  --model hifigan \
  --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/hifigan_init.pt \
  --model_dir /workspace/exp/taigi/hifigan \
  --tensorboard_dir /workspace/exp/taigi/tensorboard/hifigan \
  --ddp.dist_backend nccl \
  --num_workers 8 --prefetch 20 --pin_memory
```

GAN 第一輪不要加 `--use_amp`，先用 FP32 排除 adversarial loss、F0 與 `exp(magnitude)` 的數值不穩定。穩定後才另做 BF16 對照，不要直接覆蓋 FP32 實驗。

### 12.5 部署前只抽出 generator

訓練 checkpoint 同時包含 generator 和 discriminator；推論的 `hift.pt` 只能保存 generator。不要直接把完整 `hifigan.pt` 改名成 `hift.pt`。

```python
import torch

src = torch.load("/workspace/exp/taigi/hifigan/epoch_N_whole.pt", map_location="cpu")
dst = {
    key.removeprefix("generator."): value
    for key, value in src.items()
    if key.startswith("generator.")
}
assert dst, "No generator weights found"
torch.save(dst, "/workspace/exp/taigi/hift.pt")
```

再以 `strict=True` 載入 `CausalHiFTGenerator` 做驗證。固定用 ground-truth mel → vocoder output 比較官方與微調版；如果這一步已經有爆音、金屬音或 F0 異常，問題在 vocoder，不要用完整 TTS 掩蓋它。

### 12.6 Go/no-go 條件

只有同時滿足以下條件才採用微調 HiFT：

- Ground-truth mel reconstruction 的 MOS 明顯優於官方 `hift.pt`。
- 沒有爆音、音量抽動、高頻金屬音或 F0 跳動。
- Seen/unseen speakers 都不退化。
- 不只在降噪戲劇音軌上變好，乾淨外部測試音也沒有明顯退化。

否則部署官方 `hift.pt`。只訓 LLM 和 flow、沿用官方 vocoder 是完全合理且通常更安全的方案。

---

## 13. 模型組裝與推論

建立新的模型目錄，不覆蓋官方模型：

```text
Fun-CosyVoice3-Taigi/
├── llm.pt       微調版
├── flow.pt      官方或微調版
├── hift.pt      第一版使用官方版
├── cosyvoice3.yaml
├── CosyVoice-BlankEN/
├── campplus.onnx
└── speech_tokenizer_v3.onnx
```

華語直接模型推論：

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

model = AutoModel(model_dir="/workspace/models/Fun-CosyVoice3-Taigi")
instruction = "You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>"

for index, result in enumerate(model.inference_instruct2(
    "明天早上九點，我們一起去市場買東西。",
    instruction,
    "/workspace/data/prompts/taigi_speaker.wav",
    stream=False,
    text_frontend=False,
)):
    torchaudio.save(f"taigi_{index}.wav", result["tts_speech"], model.sample_rate)
```

臺灣台語／華語混合文字先設 `text_frontend=False`，避免既有中文 text normalization 擅自改寫台語用字。產品層應另外實作可測試的華語與台語 normalization。

---

## 14. 評估

### 14.1 分離兩種錯誤

華語輸入系統至少要分別量測：

1. 翻譯：華語語意是否正確轉成自然臺灣台語。
2. TTS：在給定正確台語稿時，發音、聲調、韻律和音質是否正確。

若只有最終音訊而沒有模型輸出的台語文字，直接模型很難判斷錯在翻譯還是發音。評測時應由母語者先聽寫生成音訊，再評語意與音韻。

### 14.2 指標

- Native-speaker MOS：自然度、台語自然性、戲劇腔偏差。
- Semantic adequacy：華語輸入和生成台語的語意一致性。
- Tone／tone-sandhi accuracy。
- 文白異讀、人名、地名、數字、日期、量詞。
- Speaker similarity：seen 和 zero-shot unseen 分開。
- 重複、漏詞、提早 EOS、無法停止。
- 音訊 clipping、DC offset、F0 discontinuity、長靜音比例。
- RTF、首包延遲、單句最大記憶體。

每次實驗固定同一批 prompt speakers、文字、random seed 與 decoding 設定，保存生成 WAV 和評分表。

---

## 15. 建議里程碑

### M0：合法性與資料驗收

- 確認授權與 Taipei-1 可用資料範圍。
- 確認 185.75 小時清洗版的 manifest、處理版本與原始檔可追溯。
- 從兩部戲分層抽樣複驗字幕語意、speaker、音樂與分離偽影。

### M1：1–5 小時 A 級 pilot

- 完成 image、manifest、Parquet 和 tiny training。
- 原始 CosyVoice3 baseline。
- 華語字幕直接到臺灣台語 LLM SFT。
- 確認 checkpoint 可重啟和部署。

### M2：20–50 小時

- 驗證文字策略、instruction、speaker split。
- 確認語意正確率達到擴大訓練門檻。
- 建立人工 `zh2taigi` test set。

### M3：全部合格資料

- 先 LLM，再依評估決定 flow。
- 預設沿用官方 `hift.pt`。
- 多 GPU／多節點 scaling 只在單節點結果可重現後進行。

### M4：通用化

- 加入合法授權的中性朗讀、新聞、日常對話與多腔口資料。
- 對戲劇域和非戲劇域分別報告指標。

---

## 16. 每次作業必須保存的 provenance

```text
git commit
image digest
base model revision/checksum
dataset manifest version
train/dev/test episode and speaker lists
retained hours by quality tier
normalization/transcription version
full YAML and launch command
GPU/node count
random seed
checkpoint and optimizer state
TensorBoard/logs
fixed evaluation outputs
```

沒有這些資訊，即使一次訓練成功，也很難在下一次 Taipei-1 allocation 重現。

---

## 17. 目前 repository 已知注意事項

- 本機 checkout 的 `third_party/Matcha-TTS` submodule 尚未初始化；image build 前必須處理。
- repository 內沒有預訓練模型，應在共享模型目錄另外準備。
- `examples/libritts/cosyvoice3/run.sh` 預設 `stop_stage=3`，不會進入 training stage 5。
- script 訊息說目前只支援 LLM training，卻又迴圈執行 `llm flow hifigan`；應分開提交三個明確作業，不使用原迴圈。
- embedding/token extraction 的部分相對路徑與執行工作目錄容易不一致，容器作業一律使用 `/opt/CosyVoice/...` 絕對路徑。
- averaging 區段的 `exp/cosyvoice` 路徑和 CosyVoice3 training output 不一致。
- DeepSpeed 範例的 optimizer learning rate 是 `0.001`，不適合直接當 SFT 的 `1e-5` 使用。先用 DDP；要用 DeepSpeed 時另外建立並審核設定。
- HiFi-GAN checkpoint 的 `hift.pt`、`hifigan.pt` 與 `generator.*` key 格式不可混用。

## 參考資料

- CosyVoice 官方 repository：https://github.com/FunAudioLLM/CosyVoice
- Fun-CosyVoice3-0.5B-2512：https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512
- CosyVoice 3 paper：https://arxiv.org/abs/2505.17589
- 《鳥來伯與十三姨》節目背景：https://zh.wikipedia.org/zh-tw/鳥來伯與十三姨
