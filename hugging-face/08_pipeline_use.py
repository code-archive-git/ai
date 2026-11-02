from transformers import pipeline

model_name = "Bllossom/llama-3.2-Korean-Bllossom-3B"

pipe = pipeline(
    "text-generation",
    model_name,
    trust_remote_code=True,
    device_map="auto"
)

result = pipe(
    "허깅 페이스 파이프라인 API 를 설명해 줘",
    max_new_tokens=128,
    max_length=None,
    temperature=0.7
)

print(result[0]['generated_text'])

