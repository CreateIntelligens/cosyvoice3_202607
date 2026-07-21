# CosyVoice3：Taipei-1 訓練與推論 Image

本專案提供單一 OCI/Docker image，同時支援資料前處理、PyTorch DDP／DeepSpeed 訓練，以及 Python、WebUI、FastAPI／gRPC 推論。建置當下的專案程式與 Matcha-TTS 會放在 `/opt/CosyVoice`，執行時不需要 `conda` 或 `conda activate`。

## 1. Image 邊界

Image 包含：

- Ubuntu 22.04、CUDA 12.4.1 development userspace、cuDNN。
- Python 3.10、PyTorch 2.3.1+cu121、torchaudio、DeepSpeed 0.15.1。
- CosyVoice 固定版本依賴、Pynini、ffmpeg、sox、ONNX Runtime GPU。
- 完整 CosyVoice 與 `third_party/Matcha-TTS` 程式碼。

Image 不包含原始 WAV、manifest、Parquet、官方模型、自訓 checkpoint、optimizer state 或 log。它們會頻繁更新且可能非常大，應放在 Taipei-1 共享儲存並掛載：

```text
/opt/CosyVoice       image 內程式碼
/workspace/models    官方模型及推論模型
/workspace/data      manifest、Parquet、測試 WAV
/workspace/exp       checkpoint、optimizer state、TensorBoard、log
```

正式作業不應另外掛載本機 repository 覆蓋 `/opt/CosyVoice`，否則 image digest 無法代表實際執行的程式。

## 2. 建置

在 repository root 執行：

```bash
git submodule update --init --recursive
test -f third_party/Matcha-TTS/matcha/__init__.py
docker info
```

使用 commit 自動命名：

```bash
bash docker/make_taipei1_image.sh
```

或直接指定 registry/tag：

```bash
bash docker/make_taipei1_image.sh \
  registry.example.org/PROJECT/cosyvoice3-taigi:cv3-074ca6dc9e80
```

Dockerfile 使用 `COPY . /opt/CosyVoice`，不是重新 clone GitHub。因此未 commit 的本地修改也會被包入；正式實驗前應 commit 並記錄 `git status --short`。`.dockerignore` 會排除模型、資料、音訊、checkpoint、Parquet 與 QC 報告。

不要用 `latest` 作正式實驗 tag。保存 image ID、digest 與 source commit：

```bash
docker image inspect IMAGE \
  --format '{{.Id}} {{json .RepoDigests}} {{json .Config.Labels}}'
```

## 3. 建置後驗證

CPU/import 測試在 build 時已執行，也可重跑：

```bash
docker run --rm IMAGE \
  python /opt/CosyVoice/tools/container_smoke_test.py --cpu-only
```

GPU 測試：

```bash
docker run --rm --gpus all --ipc=host IMAGE \
  python /opt/CosyVoice/tools/container_smoke_test.py
```

確認 `cuda_available=true`、GPU 數量正確，且 ONNX Runtime providers 包含 `CUDAExecutionProvider`。`torch_cuda_build=12.1` 是 PyTorch wheel 的 build CUDA；CUDA 12.4 host driver 可向下相容。

官方模型放在 host 後測試載入：

```bash
docker run --rm --gpus all --ipc=host \
  -v /HOST/models:/workspace/models:ro IMAGE \
  python -c "from cosyvoice.cli.cosyvoice import AutoModel; AutoModel(model_dir='/workspace/models/Fun-CosyVoice3-0.5B-2512', fp16=True); print('model-load-ok')"
```

交付前還要以一個 prompt WAV 合成一句話，並用 10–100 筆 Parquet 對 LLM 與 flow 各做一次 forward/backward，確認 checkpoint 可寫入掛載目錄並重新載入。

## 4. 8-GPU 訓練命令

LLM 範例：

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /HOST/models/Fun-CosyVoice3-0.5B-2512:/workspace/models/base:ro \
  -v /HOST/data/taigi:/workspace/data:ro \
  -v /HOST/exp/cosyvoice3-taigi:/workspace/exp \
  IMAGE \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    /opt/CosyVoice/cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --model llm \
    --config /opt/CosyVoice/examples/libritts/cosyvoice3/conf/cosyvoice3.yaml \
    --train_data /workspace/data/train.data.list \
    --cv_data /workspace/data/dev.data.list \
    --qwen_pretrain_path /workspace/models/base/CosyVoice-BlankEN \
    --onnx_path /workspace/models/base \
    --checkpoint /workspace/models/base/llm.pt \
    --model_dir /workspace/exp/llm \
    --tensorboard_dir /workspace/exp/tensorboard/llm \
    --ddp.dist_backend nccl \
    --num_workers 4 --prefetch 100 --pin_memory --use_amp
```

flow 改為 `--model flow`、`--checkpoint /workspace/models/base/flow.pt` 與獨立輸出目錄。HiFi-GAN 不可把 `hift.pt` 直接改名後傳入；仍依主訓練文件的 wrapper checkpoint 轉換流程處理。

## 5. 推論

使用 image 內的 WebUI：

```bash
docker run --rm --gpus all --ipc=host -p 50000:50000 \
  -v /HOST/models:/workspace/models:ro IMAGE \
  python /opt/CosyVoice/webui.py \
    --port 50000 \
    --model_dir /workspace/models/Fun-CosyVoice3-0.5B-2512
```

也可直接使用 `cosyvoice.cli.cosyvoice.AutoModel`，或執行 `runtime/python/fastapi`、`runtime/python/grpc`。自訓後的推論模型目錄必須包含相容的 `cosyvoice3.yaml`、`CosyVoice-BlankEN/`、`llm.pt`、`flow.pt`、`hift.pt`、ONNX tokenizer/embedding 檔及其他官方 frontend 資源。

## 6. 交付格式

優先推送到 Taipei-1 專案允許的 OCI registry：

```bash
docker login registry.example.org
docker push registry.example.org/PROJECT/cosyvoice3-taigi:cv3-074ca6dc9e80
```

若需要離線 OCI archive：

```bash
docker save IMAGE | zstd -T0 -10 -o cosyvoice3-taigi.oci.tar.zst
sha256sum cosyvoice3-taigi.oci.tar.zst > cosyvoice3-taigi.oci.tar.zst.sha256
```

若 Taipei-1 明確要求 Apptainer/SIF：

```bash
apptainer pull cosyvoice3-taigi.sif docker://REGISTRY/PROJECT/IMAGE:TAG
```

Apptainer 支援直接從 Docker/OCI registry 建立 SIF。[Apptainer pull 文件](https://apptainer.org/docs/user/main/cli/apptainer_pull.html)

若平台使用 Pyxis，結構如下；實際 allocation 與 mount 參數必須依 Taipei-1 配發文件：

```bash
srun <allocation-options> \
  --container-image=REGISTRY/PROJECT/IMAGE:TAG \
  --container-mounts=/HOST/models:/workspace/models,/HOST/data:/workspace/data,/HOST/exp:/workspace/exp \
  --container-workdir=/opt/CosyVoice \
  python /opt/CosyVoice/tools/container_smoke_test.py
```

Pyxis 可接受 OCI image URI 或 squashfs 路徑，但 Taipei-1 是否啟用該功能尚需確認。[NVIDIA Pyxis](https://github.com/NVIDIA/pyxis)

送件前向管理者確認：registry 與登入方式、compute node 能否 pull、接受 OCI／`.sqsh`／`.sif` 哪一種格式、Slurm partition/account/QoS、掛載語法、shared memory、NCCL/InfiniBand 與多節點啟動方式。
