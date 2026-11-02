from transformers import pipeline

# 정해진 몇몇 모델에 대해서만 사용가능하다.(아래 모델 입력시 에러와 함께 사용가능한 모델을 보여준다.)
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model_name = "text-classification"
classifier = pipeline(model_name, device=0)
result = classifier("I'm so upset today, and nothing is going my way.")
print(result)
