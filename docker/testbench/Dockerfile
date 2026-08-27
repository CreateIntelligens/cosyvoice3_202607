# 以正式映像為基底，只換推論引擎。CosyVoice 的程式碼、模型路徑、其餘相依都
# 跟正式環境一致，所以測出來的差異可以直接歸因到 vLLM。
FROM happy74527/cosyvoice3-taipei1:latest

# vllm 0.9.0 綁 torch 2.7.0。裝完要修兩個相依衝突，兩個都會「安靜地」壞掉：
#
#   numpy   vllm 拉 numpy 2.x，而 onnxruntime 是對 1.x 編譯的，
#           import 就炸「_ARRAY_API not found」。這個至少會中斷，看得見。
#
#   cuDNN   torch 2.7 帶 cuDNN 9，onnxruntime-gpu 1.18 要 cuDNN 8。找不到時
#           **不會報錯，直接退回 CPU**。症狀是 speaker embedding 那段從
#           0.86 秒變成 7.28 秒，總時間只快 25%，看起來像「vLLM 沒用」。
#           實際上 LLM 一直都快了 4.8 倍，是別的東西變慢了。
RUN pip install --no-cache-dir vllm==0.9.0 && \
    pip install --no-cache-dir "numpy<2" "transformers==4.51.3" \
                               "nvidia-cudnn-cu12==8.9.7.29" && \
    python -c "import torch, vllm, numpy, onnxruntime; \
               print('torch', torch.__version__, 'vllm', vllm.__version__, \
                     'numpy', numpy.__version__)"

# onnxruntime 靠這個才找得到 cuDNN 8。少了它就是上面那個安靜退回 CPU 的情況。
ENV LD_LIBRARY_PATH=/opt/cosyvoice-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
