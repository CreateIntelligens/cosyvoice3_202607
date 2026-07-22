# CosyVoice3 臺語資料準備：本機 Smoke 到 Taipei-1 全量

本流程將 drama1、drama2 的數字 speaker 目錄轉成全域唯一 ID，例如 `drama1-002`、`drama2-002`。原始資料不改名、不移動；依 `MATERIALIZE` 設定，將選中 WAV/字幕實體複製到新 workspace，或在產生 Parquet 前直接讀取來源。

### 儲存原則

- `/mnt/nas/dataset202607/segments/drama1` 與 `drama2` 只作為唯讀來源。
- 模型、smoke copy、embedding、speech token、Parquet、checkpoint、TensorBoard 都放在 repo 下的 `workspace/`。
- `workspace/` 已同時列入 `.gitignore` 與 `.dockerignore`，不會進 Git commit，也不會被打包進 image build context。
- 容器內固定將 host 的本機 workspace 掛載到 `/workspace`；不要在 NAS 建立模型 cache 或訓練輸出。

目前本機工作目錄：

```text
/home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk
```

## 1. 本次實際產物

來源：

```text
/mnt/nas/dataset202607/segments/drama1
/mnt/nas/dataset202607/segments/drama2
```

本機輸出（NAS 僅唯讀提供來源）：

```text
/home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/
└── prepared/
    ├── merged/
    │   ├── drama1-002/
    │   ├── drama1-003/
    │   ├── drama1-004/
    │   ├── drama1-005/
    │   ├── drama1-006/
    │   ├── drama2-001/
    │   ├── drama2-002/
    │   ├── drama2-003/
    │   ├── drama2-004/
    │   └── drama2-005/
    ├── data/{train,dev,test}/
    ├── selection.jsonl
    └── summary.json
```

這些是實體 copy，不是 symlink：每位 40 筆，依固定 seed 分成 30 train、5 dev、5 test。實際結果為 400 WAV、400 字幕，train/dev/test 為 300/50/50，約 104 MB。所有片段皆有 `.normalized.txt`，長度為 1.5–12 秒，並排除兩份全量 QC 報告中所有 flagged WAV。

每個 split 包含：

```text
wav.scp
text
utt2spk
spk2utt
instruct
```

`wav.scp` 使用固定容器路徑：

```text
drama1-002_002_000329_000001 /workspace/prepared/merged/drama1-002/002/002_002_000329_000001.wav
```

因此後續容器必須持續把 host workspace 掛載到 `/workspace`。

## 2. 腳本

- `tools/prepare_taigi_data.py`：掃描、QC 排除、speaker 加前綴、切分、複製與 manifest 產生。
- `tools/validate_cosyvoice_data.py`：檢查 manifest keys、speaker 反向索引、WAV、split overlap 與 Parquet rows/columns。
- `examples/taigi/cosyvoice3/prepare_data.sh`：stage 0–3 orchestration。
- `examples/taigi/cosyvoice3/train_smoke.sh`：單機或多 GPU 的 LLM/flow 訓練入口。
- `tools/make_parquet_list.py`：已修正未定義 `spk_list`，且 worker 失敗會回拋主程序。

資料準備工具預期來源布局：

```text
SOURCE_ROOT/SPEAKER/GROUP/UTTERANCE.wav
SOURCE_ROOT/SPEAKER/GROUP/UTTERANCE.normalized.txt
```

輸出目錄必須不存在，避免重跑時混入上一版本的檔案。需要重做時請改用新的 versioned 目錄，不要直接覆蓋正式資料。

## 3. 重現本次 10-speaker copy

image 重新建置並包含最新腳本後：

```bash
docker run --rm \
  -v /mnt/nas/dataset202607/segments:/datasets:ro \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -v /home/hank/Cosyvoice3/reports:/qc:ro \
  -e MODE=smoke -e STAGE=0 -e STOP_STAGE=0 \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

smoke mode 固定選擇：

```text
drama1: 002,003,004,005,006
drama2: 001,002,003,004,005
```

## 4. Stage 1–3：特徵與 Parquet

官方 base model 不屬於 smoke dataset，下載到 canonical cache：

```text
/home/hank/Cosyvoice3/workspace/models/Fun-CosyVoice3-0.5B
```

可直接使用目前 image 下載；此命令完全不掛載 NAS，並排除 smoke 訓練不需要的 1.24 GB FP32 flow 匯出檔：

```bash
docker run --rm \
  -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  -v /home/hank/Cosyvoice3/workspace/models:/models \
  IMAGE \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/models/Fun-CosyVoice3-0.5B', ignore_patterns=['flow.decoder.estimator.fp32.onnx'])"
```

本次實測下載後約 7.9 GB，並已確認 `llm.pt`、`flow.pt`、`hift.pt`、`campplus.onnx`、`speech_tokenizer_v3.onnx` 存在。若需要完整 FP32 ONNX 推論匯出，再另外下載該檔即可；PyTorch 訓練不依賴它。

embedding：

```bash
docker run --rm \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -v /home/hank/Cosyvoice3/workspace/models:/models:ro \
  -e MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
  -e MODE=smoke -e STAGE=1 -e STOP_STAGE=1 \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

speech token 需要 GPU：

```bash
docker run --rm --gpus all --ipc=host \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -v /home/hank/Cosyvoice3/workspace/models:/models:ro \
  -e MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
  -e MODE=smoke -e STAGE=2 -e STOP_STAGE=2 \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

Parquet：

```bash
docker run --rm \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -e MODE=smoke -e STAGE=3 -e STOP_STAGE=3 \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

stage 3 必須確認每個 `data.list` 指向存在的 shard，Parquet 含 `utt`、`audio_data`、`text`、`spk`、`utt_embedding`、`spk_embedding`、`speech_token`、`instruct`。

## 5. Smoke training

`train.py` 即使單 GPU 也必須由 `torchrun` 啟動。smoke 腳本預設使用 DeepSpeed ZeRO-2、BF16、CPU optimizer offload、100 frames 動態 batch，跑一個 epoch：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/torch_extensions:/tmp/cosyvoice-cache \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -v /home/hank/Cosyvoice3/workspace/models:/models:ro \
  -e MODEL_DIR=/models/Fun-CosyVoice3-0.5B \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/train_smoke.sh
```

若 H100 顯存充足而要改回一般 DDP：

```bash
TRAIN_ENGINE=torch_ddp bash /opt/CosyVoice/examples/taigi/cosyvoice3/train_smoke.sh
```

CPU offload 仍使用 GPU 做 forward/backward，只把固定占用最大的 Adam state 移到 RAM；比改造純 CPU 訓練快很多。

成功條件：

```text
/workspace/exp/llm_deepspeed/init/
/workspace/exp/llm_deepspeed/epoch_0_whole/
TRAIN/CV loss 均為有限值
程序 exit code = 0
```

只有 `init/` 不代表成功，因為它在第一個 batch 前就會寫入。可用 `MODEL=flow` 測 flow；第一輪仍不訓練 HiFi-GAN。

本機實測（2026-07-22）：

- 400 段 speaker embedding、speech token 與 Parquet 全數完成。
- 一般 DDP 在第一次 `Adam.step()` 建立 optimizer state 時 OOM；當時只有 8.47 GB 可用顯存。
- 改用 DeepSpeed ZeRO-2 CPU offload 後，506M 參數的 model-state GPU 常駐約 1.21 GB、峰值約 3.1 GB。
- 300 個 TRAIN steps 與 50 個 CV batches 全部完成，程序 exit code 為 0。
- CV loss 為 `7.5842`，沒有 NaN/Inf，`epoch_0_whole` checkpoint 成功寫出。
- 完整 DeepSpeed init 與 epoch checkpoint 約 9.5 GB；所有檔案都在 Git 忽略的本機 `workspace/`。

本次完整實測沿用官方 DeepSpeed JSON 的 `lr=0.001`；repo 內的 CPU-offload smoke 設定已改成較保守的 `1e-5`，較適合從官方 checkpoint 開始 fine-tune，正式值仍應由 dev loss 與生成品質決定。

## 6. Taipei-1 全量模式

建立新的全量 workspace，將整個目錄固定掛載到 `/workspace`，原始 segments 掛載到 `/datasets`：

```bash
srun <Taipei-1 options> \
  --container-image=IMAGE \
  --container-mounts=/SHARED/segments:/datasets:ro,/SHARED/cosyvoice3_full:/workspace,/SHARED/qc:/qc:ro \
  --container-workdir=/opt/CosyVoice \
  env MODE=full STAGE=0 STOP_STAGE=3 MATERIALIZE=copy \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

full mode 的差異：

- 掃描 drama1、drama2 全部 speaker，不限制每位筆數。
- 正式預設保留 1–30 秒片段；可用 `MIN_DURATION`、`MAX_DURATION` 覆蓋。
- speaker ID 自動變成 `drama1-NNN`／`drama2-NNN`。
- 依來源內的 GROUP（目前為 speaker 下一層目錄）切分 98%/1%/1%，避免同一集內容跨 split。
- 預設 `MATERIALIZE=copy`，建立實體合併資料；必須先估算共享儲存容量與 metadata 負載。
- 實體 copy 預設 16 個 threads，可用 `COPY_WORKERS` 依共享檔案系統負載調整。
- 若確認不需要第二份 WAV，可明確設 `MATERIALIZE=none`，manifest 直接指向 `/datasets`，Parquet 仍會內嵌音訊 bytes。
- full Parquet 預設每 shard 1000 utterances、16 processes；依 Taipei-1 CPU allocation 調整 `PARQUET_PROCESSES`。

正式建議分開提交 stage，避免長 job 失敗後全部重做：

```bash
MODE=full STAGE=0 STOP_STAGE=0 MATERIALIZE=copy ...
MODE=full STAGE=1 STOP_STAGE=1 ...
MODE=full STAGE=2 STOP_STAGE=2 ...
MODE=full STAGE=3 STOP_STAGE=3 ...
```

全量資料完成後，不要沿用 smoke 預設值直接正式訓練。H100 第一版建議使用 DDP、獨立正式 YAML、明確的 LLM/flow 作業與可續訓的單檔 checkpoint。完整命令、成功條件、DeepSpeed 限制、模型組裝與推論步驟見：

- [CosyVoice3 臺灣台語模型：完整訓練操作手冊](taigi-training-runbook.md)

正式作業應另存一份實驗 config，固定 image digest、資料 summary、Parquet data.list、seed、source commit 和 checkpoint，不要依賴可變動的 `latest` tag。
