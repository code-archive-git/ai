import torch
from transformers import AutoTokenizer,AutoConfig

from CustomClassifier import CustomClassifier

print(f'GPU 사용가능 여부 : {torch.cuda.is_available()}')
print(torch.__version__)            # 뒤에 +cpu 붙어있으면 CPU 전용 빌드

# 1단계: 커스텀 모델 클래스 만들기 - CustomClassifier

# -----------------------------------------------------------------------
# 2단계: 토크나이저와 모델 준비하기
# -----------------------------------------------------------------------
model_name = "distilbert-base-uncased" # BERT-base 모델의 핵심 지식을 압축하여 추출(증류)한 모델
# Config에 필요한 하이퍼파라미터 설정
config = AutoConfig.from_pretrained(model_name)
"""
num_labels : 분류할 클래스 개수 (긍정/부정 = 2개)
dropout_prob : 과적합(overfitting)을 막기 위한 드롭아웃 확률(의존을 막기 위한 무작위 off)
"""
config.num_labels = 2  # 예: 분류 클래스
config.classifier_dropout = 0.3

# 우리가 만든 커스텀 모델 생성 (긍정/부정 2개 클래스)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomClassifier(config)

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

# 모델 저장  - 내용 모델을 local에 저장
save_path = "./my_custom_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)