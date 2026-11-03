import torch
from transformers import AutoTokenizer, AutoModel

# 1. bert 를 기반으로 파인튜닝한 모델
model_name = "klue/bert-base" # task : fill-mask
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# 2. 문장 입력 및 토큰화
text = "파이썬 언어는 참 재미있습니다."
inputs = tokenizer(text, return_tensors="pt")

# 3. 모델 추론 (no_grad 적용)
with torch.no_grad():
    outputs = model(**inputs)

# --------------------------------------------------------------------------
# outputs에서 두 형태의 출력값 꺼내기
# --------------------------------------------------------------------------
last_hidden_state = outputs.last_hidden_state # [문장, 토큰수, 768]
pooler_output = outputs.pooler_output         # [문장, 768]

print(f"입력 문장: '{text}'")
print(f"토큰화된 토큰 수: {inputs['input_ids'].shape[1]}개\n")

print("=== 1. last_hidden_state (모든 토큰의 벡터) ===")
print(f"Shape: {last_hidden_state.shape}")
print(f"-> 설명: {last_hidden_state.shape[1]}개 토큰 각각이 768차원 벡터를 가짐\n")

print("=== 2. pooler_output (문장 대표 벡터) ===")
print(f"Shape: {pooler_output.shape}")
print(f"-> 설명: 문장 전체를 요약한 단 1개의 768차원 벡터\n")

# pooler_output 값 일부 확인
print(f"pooler_output 벡터 일부 (앞쪽 5개): {pooler_output[0, :5].tolist()}")
