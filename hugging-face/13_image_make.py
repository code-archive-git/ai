# 모든 라이브러리를 최신화 하고, 이미지생성 라이브러리인 diffusers 설치
# pip install diffusers transformers accelerate --upgrade

from diffusers import AutoPipelineForText2Image
import torch

model_id = "stabilityai/sdxl-turbo"

pipe = AutoPipelineForText2Image.from_pretrained(
    model_id,
    torch_dtype=torch.float32, # torch 크기
    variant="fp16" # 최적화된 FP16 전용 가중치 파일 사용
)

# GPU 사용
# pipe.to("cuda")

# 2. VRAM 부족을 막기 위해 CPU 오프로드 활성화 (.to("cuda") 대신 사용)
pipe.enable_model_cpu_offload()

print('이미지 생성중')
prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
image = pipe(
    prompt=prompt,
    num_inference_steps=1, # 추론단계(보통은 20~50 인데, turbo 모델은 증류 모델이므로 1~4)
    guidance_scale=0.0 # 프롬프트 가이던스(프롬프트 단어에 집착) 0~1(O) / 7.5~10(X)
).images[0]

image.save("image.png")
print('이미지 생성 및 저장 완료')