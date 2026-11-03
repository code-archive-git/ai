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
    # output_hidden_states=True 면 모든 레이어의 hidden state 를 받을 수 있음(기본)
    outputs = model(**inputs, output_hidden_states=False)

# --------------------------------------------------------------------------
# 핵심: last_hidden_state 추출
# --------------------------------------------------------------------------
# outputs.last_hidden_state 의 shape: [1, 토큰 개수, 768]
# (1: 문장 1개, 11: Special Token 포함 단어 조각 수, 768: BERT의 표현 차원)
last_hidden_state = outputs.last_hidden_state

print("=== 1. Tensor Shape 확인 ===")
print(f"last_hidden_state Shape: {last_hidden_state.shape}\n")


# --------------------------------------------------------------------------
# 활용 예시 A: 특정 단어(토큰)의 벡터 가져오기
# --------------------------------------------------------------------------
# print(f'토큰화 결과 확인 : {inputs["input_ids"][0]}')
tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
# 토큰화 결과 확인 ([CLS], 파, ##이, ##썬, 언어, ##는, 참, 재미, ##있습니다, [SEP])
print(f'토큰화 결과 확인 : {tokens}')

print("=== 2. 토큰별 벡터 살펴보기 ===")
for idx, token in enumerate(tokens):
    # 각 토큰에 해당하는 768차원의 벡터 중 앞쪽 3개 숫자만 출력해보기
    # [문장][토큰][차원0,1,2] --> [0][idx][:3] == [0, idx, :3](권장)
    vector_sample = last_hidden_state[0, idx, :3].tolist()
    print(f"토큰 [{idx}]: {token} -> 벡터 일부: {vector_sample}")
    # 768 차원에는 무엇이 담겨 있는가? 768개의 질문으로 이루어진 설문지
    # 그 단어가 가진 768가지 '의미 성분표'
    # 단어의 형태, 문법, 뉘앙스, 주제 등 단어의 '특징(Feature)'을 의미
    """
    1번 질문: 이 단어는 '명사'인가? -> 맞으면 +0.8, 아니면 -0.5
    2번 질문: '생물(사람/동물)'인가? -> 맞으면 +1.2, 아니면 -1.1
    3번 질문: 'IT/컴퓨터'와 관련된 단어인가? -> '파이썬'이면 +2.5, '사과'면 -0.3
    4번 질문: '긍정적인' 뉘앙스인가? -> '재미'면 +1.8
    """