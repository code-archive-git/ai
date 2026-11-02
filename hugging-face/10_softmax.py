# 전처리(토큰화), 모델 전달, 후처리(Softmax 연산)를 직접 해줘야 한다.
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 그 문장이 담고 있는 감정이 긍정(POSITIVE)인지 부정(NEGATIVE)인지 판별
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 1. 토크나이저 및 모델 각각 로드
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    dtype=torch.float16, # 메모리 절약(float32->float16)
    trust_remote_code=True
)

# 2. [전처리] 텍스트 -> 숫자 텐서 변환
inputs = tokenizer("Today is a great day!", return_tensors="pt")
print(f'inputs : {inputs}')
# 3. [추론] 모델 예측 실행
with torch.no_grad(): # 경사도(Gradient) 알고리즘 끔 -> 추론에서는 필요 없으므로 꺼서 자원 절약
    outputs = model(**inputs) # dict

print(outputs) # logits=tensor([[-4.3398,  4.7188]], dtype=torch.float16)
# softmax() : 를 통해 0~1 사이 값으로 변환(합은 1이 됨)
# dim = -1 : outputs 의 '원시 예측 점수(Logits - 로짓)'값들중 가장 마지막 차원의 값(막차원에 분류가 있음)
probabilities = torch.softmax(outputs.logits, dim=-1)
print(probabilities)

# argmax(probabilities): 이 중에서 가장 높은 점수를 찾는다.
# item(): tensor(1) -> 1 로 추출
pred_class = torch.argmax(probabilities).item()
print(f"예측 클래스: {pred_class}, 확률: {probabilities[0][pred_class]:.4f}")

""" 각 분류별 확률을 알고 싶다면?
prob = torch.softmax(outputs.logits,dim=-1).tolist()
print(f'probability : {prob[0]}')
print(f'POSITIVE : {prob[0][1] * 100:.2f}%')
print(f'NEGATIVE : {prob[0][0] * 100:.2f}%')
"""