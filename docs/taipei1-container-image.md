# CosyVoice3：Taipei-1 訓練與推論 Image

本文件聚焦 image 建置與交付。資料、smoke、正式 H100 訓練、checkpoint 與推論的逐步流程見 [taigi-training-runbook.md](taigi-training-runbook.md)。

本專案提供單一 OCI/Docker image，同時支援資料前處理、PyTorch DDP／DeepSpeed 訓練，以及 Python、WebUI、FastAPI／gRPC 推論。建置當下的專案程式與 Matcha-TTS 會放在 `/opt/CosyVoice`，執行時不需要 `conda` 或 `conda activate`。

## 1. Image 邊界

Image 包含：

- Ubuntu 22.04、CUDA 12.4.1 development userspace、cuDNN。
- Python 3.10、PyTorch 2.3.1+cu121、torchaudio、DeepSpeed 0.15.1。
- CosyVoice 固定版本依賴、Pynini、ffmpeg、sox、ONNX Runtime GPU。
- 完整 CosyVoice 與 `third_party/Matcha-TTS` 程式碼。

Image 不包含原始 WAV、manifest、Parquet、官方模型、自訓 checkpoint、optimizer state 或 log。它們會頻繁更新且可能非常大，應放在 Taipei-1 共享儲存並掛載：

```text
/opt/CosyVoice           image 內程式碼
/models                  外部 canonical base model，正式 job 唯讀
/workspace               外部 experiment root，不含官方 base model
/workspace/prepared      manifest、特徵、Parquet、測試 WAV
/workspace/conf          正式實驗設定
/workspace/exp           checkpoint、optimizer state
/workspace/tensorboard   log
/workspace/run-metadata  image/data/config/launch provenance
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
  registry.example.org/PROJECT/cosyvoice3-taigi:COMMIT
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
  -v /HOST/models:/models:ro IMAGE \
  python -c "from cosyvoice.cli.cosyvoice import AutoModel; AutoModel(model_dir='/models/Fun-CosyVoice3-0.5B', fp16=True); print('model-load-ok')"
```

上述內容屬於 image 發布前驗證，不會放進正式 Taipei-1 訓練 job。正式 job 只做必要的檔案、CUDA、GPU 數量與輸出權限 preflight。

## 4. 8-GPU 訓練命令

LLM 範例：

```bash
docker run --rm --gpus all --ipc=host --shm-size=32g \
  -v /HOST/taigi_v1:/workspace \
  -v /HOST/cosyvoice3/models:/models:ro \
  IMAGE \
  torchrun --standalone --nnodes=1 --nproc_per_node=8 \
    /opt/CosyVoice/cosyvoice/bin/train.py \
    --train_engine torch_ddp \
    --model llm \
    --config /workspace/conf/taigi_cosyvoice3_v1.yaml \
    --train_data /workspace/prepared/data/train/parquet/data.list \
    --cv_data /workspace/prepared/data/dev/parquet/data.list \
    --qwen_pretrain_path /models/Fun-CosyVoice3-0.5B/CosyVoice-BlankEN \
    --onnx_path /models/Fun-CosyVoice3-0.5B \
    --checkpoint /models/Fun-CosyVoice3-0.5B/llm.pt \
    --model_dir /workspace/exp/llm \
    --tensorboard_dir /workspace/tensorboard/llm \
    --ddp.dist_backend nccl \
    --num_workers 4 --prefetch 100 --pin_memory --use_amp
```

flow 改為 `--model flow`、`--checkpoint /models/Fun-CosyVoice3-0.5B/flow.pt` 與獨立輸出目錄。HiFi-GAN 不可把 `hift.pt` 直接改名後傳入；仍依主訓練文件的 wrapper checkpoint 轉換流程處理。

## 5. 推論

使用 image 內的 WebUI：

```bash
docker run --rm --gpus all --ipc=host -p 50000:50000 \
  -v /HOST/models:/models:ro IMAGE \
  python /opt/CosyVoice/webui.py \
    --port 50000 \
    --model_dir /models/Fun-CosyVoice3-0.5B
```

也可直接使用 `cosyvoice.cli.cosyvoice.AutoModel`，或執行 `runtime/python/fastapi`、`runtime/python/grpc`。自訓後的推論模型目錄必須包含相容的 `cosyvoice3.yaml`、`CosyVoice-BlankEN/`、`llm.pt`、`flow.pt`、`hift.pt`、ONNX tokenizer/embedding 檔及其他官方 frontend 資源。

## 6. 交付格式

優先推送到 Taipei-1 專案允許的 OCI registry：

```bash
docker login registry.example.org
docker push registry.example.org/PROJECT/cosyvoice3-taigi:COMMIT
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

Taipei-1 官方使用 Slurm + Pyxis + Enroot。建議先在 compute allocation 將 Docker Hub image 固定成 `.sqsh`：

```bash
srun --ntasks=1 -p REPLACE_PARTITION \
  --container-image docker.io#REPLACE_DOCKERHUB_NAMESPACE/cosyvoice3_202607:TAG \
  --container-save /mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh \
  true

srun -N 1 -p REPLACE_PARTITION --mpi=pmix \
  --gres=gpu:h100:1 --ntasks-per-node=1 \
  --container-image /mnt/home/REPLACE_USERNAME/cosyvoice3/images/cosyvoice3_TAG.sqsh \
  --container-mounts /mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1:/workspace,/mnt/home/REPLACE_USERNAME/cosyvoice3/models:/models:ro,/mnt/home/REPLACE_USERNAME/cosyvoice3/taigi_v1/cache:/tmp/cosyvoice-cache \
  bash -c 'python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"'
```

正式 PyTorch 8-GPU 作業用一個 Slurm task，再由 `torchrun --nproc_per_node=8` 管理八張 GPU。完整 `.sqsh`、互動 preflight、外部掛載與 `sbatch` 教學見 `docs/taigi-training-runbook.md`；LLM 與 HiFT 範本分別在 `examples/taigi/cosyvoice3/jobs/taipei1_train_llm.sbatch`、`examples/taigi/cosyvoice3/jobs/taipei1_train_hifigan.sbatch`。

仍需依核配資訊填入 partition/account/QoS，並確認 registry credential、資料使用政策、最長作業時間與多節點 NCCL 設定。Taipei-1 官方指南：<https://hackmd.io/@MasonWu/r1p9Hb88gg>。
