# CUDA 지원 버전 설치 - nvidia-smi
# uv pip install torch --index-url https://download.pytorch.org/whl/cu126
# window:True
# 2.14.0+cu126
import torch
print(torch.cuda.is_available())   # False면 위 원인 중 하나
print(torch.__version__)            # 뒤에 +cpu 붙어있으면 CPU 전용 빌드
# mac 버전
# print(f'mac : {torch.backends.mps.is_available()}')