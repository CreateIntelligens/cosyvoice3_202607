# CosyVoice3 臺灣台語模型：完整訓練操作手冊

本文件是可直接照順序執行的 runbook。模型設計、資料風險、華語字幕直接生成台語語音的原理，以及 HiFT／HiFi-GAN 詳細分析，請搭配 [taigi-universal-tts-training.md](taigi-universal-tts-training.md)；資料掃描與實際 smoke dataset 結果請看 [taigi-data-preparation.md](taigi-data-preparation.md)；image 建置與交付請看 [taipei1-container-image.md](taipei1-container-image.md)。

## 1. 最終目標與已驗證範圍

目標介面固定為：

```text
華語中文文字 → CosyVoice3 → 臺灣台語語音
```

不建立台文或台羅中間層。訓練 manifest 的 `text` 放華語字幕，音訊是語意對齊的臺灣台語，`instruct` 固定為：

```text
You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>
```

截至 2026-07-22 已實際驗證：

- 10 speakers、400 WAV 的資料合併、speaker 前綴與 train/dev/test 切分。
- speaker embedding、speech token 與 Parquet 產生；列數為 300/50/50。
- 一般 DDP 能完成第一個 forward/backward，但在只有 8.47 GB 可用顯存時於 Adam state 初始化 OOM。
- DeepSpeed ZeRO-2 + BF16 + CPU optimizer offload 完成 300 TRAIN steps、50 CV batches及 epoch checkpoint，exit code 為 0。
- 實測 GPU 常駐約 1.21 GB、峰值約 3.1 GB；CV loss 為 7.5842，沒有 NaN/Inf。

這證明 pipeline 能跑通，不代表 400 段 smoke checkpoint 已具備可部署品質。

## 2. 儲存與路徑規則

### 2.1 現在這台開發機

NAS 只作為原始資料的唯讀來源：

```text
/mnt/nas/dataset202607/segments/drama1
/mnt/nas/dataset202607/segments/drama2
```

所有可重建產物放在 repo 下，且已被 Git 與 Docker build context 忽略：

```text
/home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/
├── models/Fun-CosyVoice3-0.5B/
├── prepared/
├── conf/
├── exp/
├── tensorboard/
└── torch_extensions/
```

不要在 NAS 下載模型、建立 cache、Parquet、checkpoint 或 TensorBoard log。

### 2.2 Taipei-1

Taipei-1 的 `/workspace` 是「容器內掛載點」；host 端應使用官方提供的持久儲存，例如個人 `/mnt/home/<USERNAME>` 或團隊 `/mnt/shared/<ORG_NAME>`，不是目前這台機器的 NAS：

```text
/workspace/
├── models/Fun-CosyVoice3-0.5B/
├── prepared/data/{train,dev,test}/
├── conf/
├── exp/
├── tensorboard/
└── run-metadata/
```

image 內程式固定在 `/opt/CosyVoice`。正式作業不要用 host repo 覆蓋它，確保 image digest 能代表實際程式碼。

### 2.3 容器掛載契約

容器外的實體路徑可以因機器而不同，但容器內路徑必須固定。建議契約：

| 內容 | 容器內固定路徑 | 權限 | 是否可能寫進 manifest/checkpoint |
|---|---|---|---|
| Image 內程式 | `/opt/CosyVoice` | image 內唯讀 | 會出現在 launch command |
| 實驗 workspace | `/workspace` | 可寫 | 會 |
| 原始 drama1/drama2 | `/datasets` | 唯讀 | `MATERIALIZE=none` 時會 |
| QC JSON | `/qc` | 唯讀 | 通常不會，但 summary 會記來源 |
| PyTorch/DeepSpeed extension cache | `/tmp/cosyvoice-cache` | 可寫 | 不會 |

以下三台機器可以使用完全不同的 host path：

```text
開發機 host: /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk
Taipei-1 host: /mnt/home/<USERNAME>/cosyvoice3/taigi_v1
其他 GPU host: /srv/training/cosyvoice3/taigi_v1
```

只要都掛成容器內 `/workspace`，manifest 與命令就不需要改：

```text
HOST 任意實體路徑 → /workspace
HOST 原始資料路徑 → /datasets
HOST QC 路徑       → /qc
```

### 2.4 哪些檔案保存絕對路徑

目前 pipeline 會在下列檔案保存容器內絕對路徑：

- `wav.scp`：WAV 的 `/workspace/...` 或 `/datasets/...`。
- Parquet `data.list`：每個 shard 的 `/workspace/...`。
- `utt2data.list`／`spk2data.list` JSON：Parquet shard 路徑。
- Parquet 的 `wav` 欄位：來源 WAV 路徑。
- 訓練 YAML、log、DeepSpeed client state：可能保存 model/output 路徑。

因此：

1. 不要把 `/home/hank/...`、`/mnt/nas/...` 或 Taipei-1 host 實體路徑直接寫進 manifest。
2. 產生資料時就使用固定容器路徑。
3. 搬到另一台機器時，改 mount 左側 host path，不改右側 container path。
4. 不要對數十萬行 manifest 做未驗證的全域字串取代；路徑契約變更時應重新產生並跑 validator。

`prepare_taigi_data.py` 的 `--container-source` 與 `--container-output-root` 就是用來區分「掃描時看見的 host/mount 路徑」和「寫進 manifest 的固定容器路徑」。既有 orchestration 已將它們設為 `/datasets` 與 `/workspace/prepared`。

### 2.5 推薦：單一外部 experiment root

最穩定的做法是把模型、prepared data、設定、checkpoint 與 log 放在同一個容器外目錄：

```text
/mnt/home/<USERNAME>/cosyvoice3/taigi_v1/
├── models/
├── prepared/
├── conf/
├── exp/
├── tensorboard/
├── run-metadata/
└── cache/
```

Docker：

```bash
task_workspace=/SHARED/PROJECT/cosyvoice3/taigi_v1
task_datasets=/SHARED/DATASETS/drama_segments
task_qc=/SHARED/PROJECT/cosyvoice3/qc

docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v "${task_workspace}:/workspace" \
  -v "${task_datasets}:/datasets:ro" \
  -v "${task_qc}:/qc:ro" \
  IMAGE \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

Taipei-1 官方 Pyxis/Enroot 掛載結構：

```bash
task_workspace=/mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1
task_datasets=/mnt/shared/REPLACE_ORG/drama_segments
task_qc=/mnt/home/REPLACE_USERNAME/cosyvoice3/qc
task_cache=/mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1/cache

srun -N 1 -p REPLACE_PARTITION --ntasks=1 \
  --container-image /mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh \
  --container-mounts "${task_workspace}:/workspace,${task_datasets}:/datasets,${task_qc}:/qc,${task_cache}:/tmp/cosyvoice-cache" \
  bash -c 'cd /opt/CosyVoice && bash examples/taigi/cosyvoice3/prepare_data.sh'
```

官方文件確認 `--container-mounts HOST:CONTAINER` 與逗號分隔多掛載。文件沒有示範 `:ro` 第三欄，因此上面的正式相容指令只使用兩欄；原始資料仍應以 Taipei-1 ACL／權限設成不可寫，並由腳本遵守不寫回 `/datasets`、`/qc`。若管理者確認該版 Pyxis 支援 `:ro`，再加上唯讀旗標。

### 2.6 Taipei-1 官方流程：Docker Hub 到 `.sqsh`

Taipei-1 已整合 Slurm、Pyxis 與 Enroot，不能在叢集上使用 `docker build`。推薦流程是：在開發機建置並推送 Docker/OCI image，再由 Taipei-1 compute allocation 拉取並保存成固定 `.sqsh`。

開發機：

```bash
bash docker/make_taipei1_image.sh docker.io/REPLACE_DOCKERHUB_NAMESPACE/cosyvoice3_202607:TAG
docker login docker.io
docker push docker.io/REPLACE_DOCKERHUB_NAMESPACE/cosyvoice3_202607:TAG
```

Taipei-1 login node 只建立小目錄與提交作業，不在 login node 下載模型、轉資料或編譯：

```bash
module load slurm
mkdir -p /mnt/home/REPLACE_USERNAME/cosyvoice3/images
mkdir -p /mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1/{models,prepared,conf,exp,tensorboard,run-metadata,cache}
```

透過 compute node 將 registry image 固定成 `.sqsh`：

```bash
srun --ntasks=1 -p REPLACE_PARTITION \
  --container-image docker.io#REPLACE_DOCKERHUB_NAMESPACE/cosyvoice3_202607:TAG \
  --container-save /mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh \
  true
```

`docker.io#...` 是依官方文件的 registry URI 規則（registry 後以 `#` 取代 `/`）套用到 Docker Hub。先用一個短作業驗證；私有 image 需依官方方式把 registry credential 設在 `~/.config/enroot/.credentials`，不要把密碼寫進 job script。正式訓練固定使用 `.sqsh` 絕對路徑，避免 tag 漂移或每個 job 重拉 image。

互動式 preflight：

```bash
srun -N 1 -p REPLACE_PARTITION --mpi=pmix \
  --gres=gpu:h100:1 --ntasks-per-node=1 \
  --container-image /mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh \
  --container-mounts /mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1:/workspace,/mnt/shared/REPLACE_ORG/drama_segments:/datasets,/mnt/home/REPLACE_USERNAME/cosyvoice3/qc:/qc,/mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1/cache:/tmp/cosyvoice-cache \
  --pty /bin/bash

cd /opt/CosyVoice
python tools/container_smoke_test.py --cpu-only
nvidia-smi
```

正式 immutable job 不需要 `--container-writable` 或 `--container-remap-root`；所有寫入都導向 `/workspace` 與 `/tmp/cosyvoice-cache`。只有為了暫時修改容器內容做診斷時才使用這兩個選項，修改後要用 `enroot export` 另存新 image，不要把互動修改當成正式可重現環境。`--container-mount-home` 也是選用，這套流程已顯式掛載所需路徑，不依賴容器內的 home。

> Taipei-1 官方指南明確限制機敏、營業秘密或專有資訊，也禁止未授權／違法資料。三立與《鳥來伯與十三姨》的音影音源、字幕及衍生 Parquet 在上傳前，必須先取得資料授權並向 Taipei-1 管理單位書面確認符合使用政策；技術上可掛載不代表政策上允許處理。

### 2.7 替代：多個外部目錄分開掛載

若模型、資料與輸出由不同儲存區管理，也可分開掛載：

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /HOST/base-model:/workspace/models/Fun-CosyVoice3-0.5B:ro \
  -v /HOST/prepared:/workspace/prepared:ro \
  -v /HOST/conf:/workspace/conf:ro \
  -v /HOST/exp:/workspace/exp \
  -v /HOST/tensorboard:/workspace/tensorboard \
  -v /HOST/jobs:/workspace/jobs:ro \
  IMAGE \
  bash /workspace/jobs/train_llm_v1.sh
```

注意：

- Docker 通常支援這種 nested mounts；Taipei-1 的 Pyxis/Enroot 是否允許及掛載順序要先做 tiny test。
- 訓練時 `prepared`、base model、conf 可唯讀；`exp`、TensorBoard 與 run metadata 必須可寫。
- 若 `data.list` 內容仍是 `/workspace/prepared/...`，掛載點就必須維持相同。
- 多個掛載比單一 experiment root 更容易有權限與遺漏問題，第一版優先使用單一 root。

### 2.8 `MATERIALIZE=copy` 與 `none`

`MATERIALIZE=copy`：

- 選中的 WAV/字幕複製到 `/workspace/prepared/merged`。
- `wav.scp` 指向 `/workspace/prepared/merged/...`。
- 後續重跑 validator、embedding 或 token extraction 不需要再次讀原始 `/datasets`。
- 需要額外容量與大量 metadata I/O，但最容易搬移與重現。

`MATERIALIZE=none`：

- 不複製 WAV，`wav.scp` 直接指向 `/datasets/...`。
- Stage 1/2 及帶 `--check-audio` 的 validator 必須繼續掛載相同的 `/datasets`。
- Stage 3 Parquet 會內嵌 `audio_data`，一般訓練主要讀 Parquet bytes，但仍保留 `wav` 欄位；不要因此假設所有後續工具都不需要來源 mount。
- 原始資料位置或 mount contract 改變時，manifest 也會失效。

185.75 小時正式版本若共享空間足夠，第一版建議 `copy`。確認容量不足且 `/datasets` 能長期穩定掛載後，才選 `none`。

### 2.9 Model 放在容器外

模型不應烘進 image。可以放在 experiment root：

```text
HOST: /SHARED/PROJECT/cosyvoice3/taigi_v1/models/Fun-CosyVoice3-0.5B
CONTAINER: /workspace/models/Fun-CosyVoice3-0.5B
```

也可用獨立唯讀模型庫：

```bash
-v /SHARED/MODELS/Fun-CosyVoice3-0.5B:/workspace/models/Fun-CosyVoice3-0.5B:ro
```

checkpoint 必須寫到另一個可寫路徑，例如 `/workspace/exp`；不要將訓練輸出寫回 base model mount。

### 2.10 權限與 root-owned 檔案

容器預設 root 可能在 CIFS/NAS 建立 host 使用者無法刪除的檔案。本專案已發生過這種情況，因此：

- 原始 `/datasets` 與 `/qc` 一律唯讀掛載。
- 不直接把 NAS 目錄當 `/workspace`。
- 作業前由平台允許的帳號建立 workspace 並確認 UID/GID、quota 與 ACL。
- Docker 若要使用 `--user UID:GID`，先確認 CUDA、共享記憶體、cache 與 image 內檔案皆可讀；做一次完整 smoke 後才正式採用。
- Taipei-1 使用平台提供的身份映射，不自行在 compute node `chown` 大範圍共享目錄。

### 2.11 掛載後 preflight

每個 job 在執行資料處理或訓練前先檢查：

```bash
set -euo pipefail
mkdir -p /workspace/exp /workspace/tensorboard /workspace/run-metadata
test -r /opt/CosyVoice/cosyvoice/bin/train.py
test -r /workspace/models/Fun-CosyVoice3-0.5B/llm.pt
test -r /workspace/prepared/data/train/parquet/data.list
test -r /workspace/prepared/data/dev/parquet/data.list
test -w /workspace/exp
df -h /workspace

python -c "from pathlib import Path; p=Path('/workspace/prepared/data/train/parquet/data.list'); q=Path(p.read_text().splitlines()[0]); print(q); assert q.is_file()"
```

Stage 0/1/2 或 `MATERIALIZE=none` 再加：

```bash
test -r /datasets/drama1
test -r /datasets/drama2
```

若 preflight 顯示 `data.list` 指向不存在的 host 路徑，先修正 mount contract 或重建 manifest，不要開始昂貴的 H100 job。

## 3. 訓練前必備項目

必須先準備：

- 已確認可用於訓練與指定運算平台的資料權利。
- 固定 Git commit、乾淨或已記錄的 worktree、固定 image tag/digest。
- 官方 `FunAudioLLM/Fun-CosyVoice3-0.5B-2512` base model。
- train/dev/test manifest、QC 報告及 dataset summary。
- 至少一組固定的華語輸入、台語參考語意與 prompt speaker WAV。
- 足夠的 checkpoint 空間；單次 DeepSpeed smoke 的 init + epoch checkpoint 約 9.5 GB。

正式訓練前記錄：

```bash
git rev-parse HEAD
git status --short
docker image inspect IMAGE --format '{{.Id}} {{json .RepoDigests}} {{json .Config.Labels}}'
```

## 4. 建置並驗證 image

在 repo root：

```bash
git submodule update --init --recursive
test -f third_party/Matcha-TTS/matcha/__init__.py
bash docker/make_taipei1_image.sh REGISTRY/PROJECT/cosyvoice3-taigi:COMMIT
```

CPU 與 GPU 驗證：

```bash
docker run --rm IMAGE \
  python /opt/CosyVoice/tools/container_smoke_test.py --cpu-only

docker run --rm --gpus all --ipc=host IMAGE \
  python /opt/CosyVoice/tools/container_smoke_test.py
```

成功條件：PyTorch、torchaudio、DeepSpeed、CosyVoice、Matcha-TTS 可 import；`torch.cuda.is_available()` 為 true；ONNX Runtime providers 包含 `CUDAExecutionProvider`。

目前 image 尚未重建時，可將 repo 唯讀掛到 `/project` 測試，但必須同時設定：

```bash
-e COSYVOICE_ROOT=/project \
-e PYTHONPATH=/project:/project/third_party/Matcha-TTS \
-v /home/hank/Cosyvoice3:/project:ro
```

這只適合本機開發，不是正式 Taipei-1 作業方式。

## 5. 準備官方模型

在可連網的開發或 login 節點下載，不假設 compute node 能連網：

```bash
docker run --rm \
  -v /HOST/WORKSPACE:/workspace \
  IMAGE \
  python -c "from huggingface_hub import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir='/workspace/models/Fun-CosyVoice3-0.5B', ignore_patterns=['flow.decoder.estimator.fp32.onnx'])"
```

訓練必要檔：

```text
cosyvoice3.yaml
CosyVoice-BlankEN/config.json
campplus.onnx
speech_tokenizer_v3.onnx
llm.pt
flow.pt
hift.pt
```

逐一用 `test -s` 驗證，不要只檢查目錄存在。`flow.decoder.estimator.fp32.onnx` 是大型推論匯出檔，PyTorch 訓練不依賴它。

## 6. 資料準備

### 6.1 本機 smoke

現有 smoke 資料位於：

```text
/home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/prepared
```

驗證：

```bash
docker run --rm \
  -v /home/hank/Cosyvoice3:/project:ro \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace:ro \
  IMAGE \
  python /project/tools/validate_cosyvoice_data.py \
    --prepared-dir /workspace/prepared --check-audio --require-parquet
```

### 6.2 Taipei-1 全量資料

將來源唯讀掛成 `/datasets`，QC 報告唯讀掛成 `/qc`，輸出 workspace 可寫。四個 stage 建議分開提交：

```bash
MODE=full STAGE=0 STOP_STAGE=0 MATERIALIZE=copy \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh

MODE=full STAGE=1 STOP_STAGE=1 NUM_THREADS=16 \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh

MODE=full STAGE=2 STOP_STAGE=2 NUM_THREADS=16 \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh

MODE=full STAGE=3 STOP_STAGE=3 \
  PARQUET_PROCESSES=16 UTTS_PER_PARQUET=1000 \
  bash /opt/CosyVoice/examples/taigi/cosyvoice3/prepare_data.sh
```

各 stage 功能：

1. Stage 0：speaker ID 加 `drama1-`／`drama2-` 前綴、切分、manifest、選擇性 copy。
2. Stage 1：以 `campplus.onnx` 產生 utterance/speaker embedding。
3. Stage 2：以 `speech_tokenizer_v3.onnx` 產生 speech token，需要 GPU。
4. Stage 3：內嵌音訊與特徵，產生 Parquet 與 `data.list`。

全量資料完成後一定要執行：

```bash
python /opt/CosyVoice/tools/validate_cosyvoice_data.py \
  --prepared-dir /workspace/prepared --check-audio --require-parquet
```

成功條件：manifest keys 一致、WAV 可解碼、split 無 utt overlap、Parquet rows 等於 manifest rows，且包含：

```text
utt audio_data wav text spk utt_embedding spk_embedding speech_token instruct
```

## 7. 建立正式實驗設定

不要直接修改官方 YAML：

```bash
mkdir -p /workspace/conf
cp /opt/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
  /workspace/conf/taigi_cosyvoice3_v1.yaml
```

第一輪建議值：

```yaml
batch:
    batch_type: dynamic
    max_frames_in_batch: 800

train_conf:
    optim: adam
    optim_conf:
        lr: 1.0e-5
    scheduler: constantlr
    max_epoch: 20
    grad_clip: 5
    accum_grad: 2
    log_interval: 20
    save_per_step: 1000
```

先從每 rank 800 frames 起跑，再依 H100 實測顯存提高到 1200、1600 或 2000。調整 batch 時同時記錄 GPU 數量與 gradient accumulation，避免 global batch 無意間放大。

保存 YAML 副本及 checksum：

```bash
sha256sum /workspace/conf/taigi_cosyvoice3_v1.yaml \
  > /workspace/run-metadata/taigi_cosyvoice3_v1.yaml.sha256
```

## 8. 本機低顯存 smoke

腳本預設使用經實測的 DeepSpeed ZeRO-2 CPU offload：

```bash
docker run --rm --gpus all --ipc=host --shm-size=16g \
  -e COSYVOICE_ROOT=/project \
  -e PYTHONPATH=/project:/project/third_party/Matcha-TTS \
  -v /home/hank/Cosyvoice3:/project:ro \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk:/workspace \
  -v /home/hank/Cosyvoice3/workspace/cosyvoice3_smoke_10spk/torch_extensions:/tmp/cosyvoice-cache \
  IMAGE \
  bash /project/examples/taigi/cosyvoice3/train_smoke.sh
```

設定檔是：

```text
examples/taigi/cosyvoice3/conf/ds_stage2_cpu_offload.json
```

它使用 BF16、ZeRO-2、CPU AdamW、50M communication buckets，repo 版本的 learning rate 為 `1e-5`。主機建議至少有 16 GB 可用 RAM；本次 DeepSpeed 估算 CPU model-state 需求約 11.31 GB。

## 9. Taipei-1/H100 正式 LLM 訓練

H100 80 GB 第一選擇是一般 DDP：格式簡單、checkpoint 是單一 `.pt`，目前程式也能用 `--checkpoint` 直接續跑。先單節點 8 GPU，確認穩定後才擴多節點。

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /opt/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model llm \
  --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
  --train_data /workspace/prepared/data/train/parquet/data.list \
  --cv_data /workspace/prepared/data/dev/parquet/data.list \
  --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
  --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/llm.pt \
  --model_dir /workspace/exp/taigi_v1/llm_ddp \
  --tensorboard_dir /workspace/tensorboard/taigi_v1/llm_ddp \
  --ddp.dist_backend nccl \
  --num_workers 4 --prefetch 20 --pin_memory --use_amp
```

Slurm/Pyxis 正式單節點範本已放在 `examples/taigi/cosyvoice3/jobs/taipei1_train_llm.sbatch`。提交方式：

```bash
module load slurm

sbatch -p REPLACE_PARTITION -A REPLACE_ACCOUNT \
  --export=ALL,TP1_IMAGE=/mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh,TP1_WORKSPACE=/mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1,TP1_DATASETS=/mnt/shared/REPLACE_ORG/drama_segments,TP1_QC=/mnt/home/REPLACE_USERNAME/cosyvoice3/qc \
  examples/taigi/cosyvoice3/jobs/taipei1_train_llm.sbatch
```

PyTorch 一個 node 只配置 `--ntasks-per-node=1`，因為 `torchrun --nproc_per_node=8` 會自行建立八個 GPU worker；若同時配置八個 Slurm tasks，可能意外啟動八組 `torchrun`。`ACCOUNT`、partition、QoS 與最長 wall time 必須使用 Taipei-1 實際配發資訊，不要照抄 placeholder。先用單節點 8×H100；多節點還需依實際節點清單設定 rendezvous，不能只把 `--nodes` 改大。

### 9.1 成功判準

第一個正式 job 至少確認：

- 每個 rank 都加入 NCCL，沒有 silent hang。
- step 持續增加，TRAIN/CV loss、grad norm 有限。
- GPU memory、utilization 與 data loading 沒有長時間飢餓。
- `epoch_0_whole.pt` 存在且不是只有 `init.pt`。
- 用該 checkpoint 組裝模型後能合成固定評估句。

### 9.2 DDP 續訓

將 `--checkpoint` 改為上一個完整 checkpoint：

```text
/workspace/exp/taigi_v1/llm_ddp/epoch_N_whole.pt
```

該檔含 `epoch` 與 `step`。續訓前確認 YAML、資料與 GPU world size 是否和原實驗一致，並將 stdout/stderr 保存到新 log。

## 10. DeepSpeed checkpoint 的限制與部署轉換

低顯存模式會輸出目錄式 checkpoint：

```text
exp/llm_deepspeed/
├── init/
├── epoch_0_whole/
└── zero_to_fp32.py
```

目前 `train.py --checkpoint` 在 DeepSpeed 初始化前使用 `torch.load()`，不能直接把上述目錄當成 optimizer-resume checkpoint。也就是：能保存 DeepSpeed state，但未完成正式的 DeepSpeed 中斷續訓整合。會遇到 job requeue/preemption 的正式長訓練，優先使用 DDP，或先補上 `model.load_checkpoint()` 流程並另外做 restart smoke test。

若只要部署模型權重，在含 DeepSpeed 的 image 內轉成單檔：

```bash
python /workspace/exp/llm_deepspeed/zero_to_fp32.py \
  /workspace/exp/llm_deepspeed/epoch_0_whole \
  /workspace/exp/llm_deepspeed/llm_fp32.pt
```

轉換需要足夠 RAM 與額外磁碟空間。完成後以 `torch.load(..., map_location='cpu')` 檢查 key 數量，再載入 CosyVoice3 LLM 驗證。

## 11. Checkpoint 選擇與平均

不要只選 train loss 最低者。先依 dev loss 排序，再用固定華語評估句聽語意、台語自然度、重複／漏詞與音色。

DDP checkpoint 可平均：

```bash
python /opt/CosyVoice/cosyvoice/bin/average_model.py \
  --dst_model /workspace/exp/taigi_v1/llm_avg5.pt \
  --src_path /workspace/exp/taigi_v1/llm_ddp \
  --num 5 --val_best
```

先確認 `average_model.py` 實際找到的檔案清單，不要沿用官方 `run.sh` 中錯誤的 `exp/cosyvoice` 路徑。

## 12. Flow 訓練

先評估只換 LLM 的結果。如果內容語意正確，但聲調、節奏或台語聲學表現仍不理想，再從官方 `flow.pt` 微調：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 \
  /opt/CosyVoice/cosyvoice/bin/train.py \
  --train_engine torch_ddp \
  --model flow \
  --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
  --train_data /workspace/prepared/data/train/parquet/data.list \
  --cv_data /workspace/prepared/data/dev/parquet/data.list \
  --qwen_pretrain_path /workspace/models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
  --onnx_path /workspace/models/Fun-CosyVoice3-0.5B \
  --checkpoint /workspace/models/Fun-CosyVoice3-0.5B/flow.pt \
  --model_dir /workspace/exp/taigi_v1/flow_ddp \
  --tensorboard_dir /workspace/tensorboard/taigi_v1/flow_ddp \
  --ddp.dist_backend nccl \
  --num_workers 4 --prefetch 20 --pin_memory --use_amp
```

LLM 與 flow 分開作業、分開 output directory，不使用官方 recipe 中同時迴圈訓練多個 model 的寫法。

## 13. HiFT／HiFi-GAN 決策

HiFT fine-tuning 排入正式實驗，但官方 `hift.pt` 永遠保留為 baseline；candidate 只有通過 vocoder-only 與 end-to-end A/B gate 才部署。官方協作者表示通常不需訓練 HiFi-GAN，因此不能把「完成訓練」本身當作採用理由。

執行順序：

1. 複製／凍結官方 `hift.pt`，記錄 SHA-256。
2. 用 10–20 小時 A 級音訊、FP32、小 learning rate 短跑，不從隨機權重開始。
3. 用 `tools/convert_hift_checkpoint.py wrap` 建立 `generator.*` wrapper；`train.py` 會強制 generator coverage 100%。
4. 先做 ground-truth mel/F0 → waveform A/B；不通過就停止。
5. 有改善才擴大資料或 epoch，並監控爆音、金屬音、F0、clipping、RMS/LUFS、長靜音及 CER。
6. 用 `tools/convert_hift_checkpoint.py extract` 只抽出 generator，另存 candidate `hift.pt`。
7. 固定同一組 LLM/flow 做 end-to-end 盲測；不佳就直接選官方 baseline。

注意：HiFi-GAN 目前只能用 `torch_ddp`，程式明確不支援 DeepSpeed 包裝 generator＋discriminator；GAN 的 `accum_grad` 必須為 1。Taipei-1 保守單 GPU 範本是：

```text
examples/taigi/cosyvoice3/jobs/taipei1_train_hifigan.sbatch
```

完整命令、保守設定、轉換與 go/no-go 條件見總手冊第 12 節。

## 14. 組裝可推論模型

不要覆蓋官方 base：

```bash
cp -a /workspace/models/Fun-CosyVoice3-0.5B \
  /workspace/models/Fun-CosyVoice3-Taigi-v1

cp /workspace/exp/taigi_v1/llm_avg5.pt \
  /workspace/models/Fun-CosyVoice3-Taigi-v1/llm.pt

# 只有 flow 通過評估時才替換。
cp /workspace/exp/taigi_v1/flow_ddp/FLOW_CHECKPOINT.pt \
  /workspace/models/Fun-CosyVoice3-Taigi-v1/flow.pt
```

預設保留 base 目錄中的官方 `hift.pt`；只有候選模型通過第 13 節全部 gate 才替換，而且另建模型目錄，不覆蓋官方 base。

推論 smoke：

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

model = AutoModel(model_dir="/workspace/models/Fun-CosyVoice3-Taigi-v1")
instruction = "You are a helpful assistant. 請將輸入的華語翻譯成自然的臺灣台語並朗讀。<|endofprompt|>"

for i, result in enumerate(model.inference_instruct2(
    "明天早上九點，我們一起去市場買東西。",
    instruction,
    "/workspace/eval/prompt_speaker.wav",
    stream=False,
    text_frontend=False,
)):
    torchaudio.save(f"/workspace/eval/output_{i}.wav", result["tts_speech"], model.sample_rate)
```

訓練與推論 instruction 必須完全相同，包含 `<|endofprompt|>`。

## 15. 評估與放行

固定測試集至少包含：

- Seen speakers、unseen episodes。
- 完全 unseen speakers。
- 非戲劇語域：新聞、生活、說明文。
- 數字、日期、時間、地址、貨幣、人名、地名。
- 華語夾英文、否定句、疑問句與長句。

每個 checkpoint 分開評：

- 華語語意是否正確轉成自然台語。
- 台語發音、聲調、tone sandhi、節奏。
- 重複、漏詞、提早停止、長靜音。
- Speaker similarity、自然度、戲劇腔偏差。
- clipping、爆音、金屬音、F0 跳動。

若 LLM 內容錯誤，先處理文字映射或資料；若內容正確但聲學不自然，再考慮 flow。HiFT 短跑 candidate 可以照第 13 節執行，但只有 vocoder-only A/B 顯示優於官方 baseline 才擴大全量訓練。

## 16. 每次正式作業保存內容

```text
git commit 與 git status
image tag、ID、digest
base model revision/checksum
dataset summary、selection、split 與 data.list checksum
完整 YAML、DeepSpeed JSON、launch command
Slurm job ID、node/GPU/world size
stdout/stderr、TensorBoard
checkpoint、optimizer state、轉檔紀錄
固定評估文字、prompt WAV、生成 WAV、評分表
```

建議每次建立 `/workspace/run-metadata/EXPERIMENT_ID/`，不要只靠 shell history。

## 17. 常見失敗排查

### 一開始就 OOM

- 確認是否在 Adam state 初始化，而不是 forward。
- 降 batch 只會降低 activation，無法消除固定 optimizer state。
- 本機改用 DeepSpeed CPU offload；H100 正式訓練確認是否有其他程序占 GPU。

### DeepSpeed 出現 ConstantLR warmup_steps 錯誤

必須使用本 repo 已修正的 `cosyvoice/utils/train_utils.py`。若掛載新版 repo 測舊 image，要設定 `/project` 優先的 `PYTHONPATH`。

### 只有 init checkpoint

`init` 在第一個 batch 前就會產生，不代表成功。必須看到 step 增加、CV 完成與 `epoch_N_whole`。

### Parquet 顯示 element 欄位

list 的子欄位可名為 `element`；頂層必須仍是 `utt_embedding`、`spk_embedding`、`speech_token`。使用本 repo 的 validator 檢查頂層 Arrow schema。

### HiFi-GAN 訓練後聲音損壞

先確認 generator prefix、載入覆蓋率、部署時是否只抽 generator，再檢查 GAN learning rate、AMP、F0、sample rate、裁切與降噪偽影。

## 18. 正式開跑前最終清單

- [ ] image 已包含目前 repo 修改且通過 CPU/GPU smoke。
- [ ] 模型與所有 workspace 產物不在 NAS、不進 Git、不烘進 image。
- [ ] 全量 manifest/Parquet validator 通過。
- [ ] train/dev/test 以 group/episode 隔離，speaker 分布已保存。
- [ ] 華語 `text` 與固定 instruction 已抽查。
- [ ] LLM、flow 分開訓練；HiFT 保留官方 baseline 並另跑短程 candidate，通過雙重 A/B gate 才採用。
- [ ] H100 先單節點 DDP，再評估多節點。
- [ ] checkpoint、log、metadata 路徑容量足夠。
- [ ] 固定 inference 與人工評估集已準備。
- [ ] 已確認 Taipei-1 account、partition、QoS、registry、mount 與最長作業時間。
