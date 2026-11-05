import torch
from transformers import AutoTokenizer

from CustomClassifier import CustomClassifier

# -----------------------------------------------------------------------
# 2단계: 토크나이저와 모델 준비하기
# -----------------------------------------------------------------------
# 1. 저장된 모델 디렉토리 경로
save_path = "./my_custom_model"
print("모델과 토크나이저를 불러오는 중...")
# 2. 저장된 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained(save_path)
# 3. 저장된 설정(Config) 및 모델 로드
model = CustomClassifier.from_pretrained(save_path)

# -----------------------------------------------------------------------
# 3단계: 예제 문장으로 실제 동작 확인해보기 (추론, Inference)
# -----------------------------------------------------------------------
sentences = [
    "I really loved this movie, it was fantastic!",
    "This was a waste of time, I hated it.",
]

# 문장을 토큰화 (padding=True: 길이를 맞춰줌, truncation=True: 너무 길면 자름, max_length:기본 512)
inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")

# 모델은 학습 없이 결과만 볼 때는 eval 모드로 바꿔주는 것이 좋습니다.
model.eval()

# torch.no_grad(): 추론할 때는 기울기(gradient)를 계산할 필요가 없으므로
# 메모리와 연산을 절약해줍니다.
with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

print("모델 출력 (logits):\n", logits)

# logits는 아직 "확률"이 아닌 원시 점수입니다.
# softmax를 적용하면 각 클래스에 대한 확률로 바꿀 수 있습니다.
# dim=0 세로 방향을 따라 연산
# dim=1/-1 가로 방향을 따라 이동
probs = torch.softmax(logits, dim=-1)
print("\n확률로 변환 (softmax):\n", probs)

# 가장 확률이 높은 클래스를 예측 결과로 선택
predictions = torch.argmax(probs, dim=-1)
print("\n예측 클래스 (0=부정, 1=긍정):\n", predictions)