from transformers import pipeline

# device = 0 : GPU
# device = -1 : CPU(default)
# device = "cuda" : 사용 가능한 GPU
summ = pipeline("summarization", model="EbanLee/kobart-summary-v3", device=-1)

text = """자연어 처리(NLP)는 컴퓨터가 인간의 언어를 이해하고 생성할 수 있도록 하는 기술입니다.
딥러닝의 발전과 함께 다양한 모델들이 제안되었으며, 실제 서비스에도 널리 활용되고 있습니다."""

# 요약 내용은 10~30자 사이로...
result = summ(text,max_length=20, min_length=10, do_sample=False)
print(result[0]["summary_text"])