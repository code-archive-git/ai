import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, AutoConfig

class CustomClassifier(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.backbone = AutoModel.from_config(config)
        num_labels = getattr(config, "num_labels", 2)  # config에 설정된 num_labels 사용 (기본값 2)
        dropout_prob = getattr(config, "classifier_dropout", 0.3)  # 드롭아웃 비율

        hidden_size = config.hidden_size

        self.custom_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384로 차원 축소
            nn.ReLU(),  # 비선형성 추가
            nn.Dropout(dropout_prob),  # 과적합 방지(무작위 30% 끄기)
            nn.Linear(hidden_size // 2, num_labels),  # 384 -> 2 (최종 클래스 개수)
        )

        self.post_init()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        print(f'shape : {outputs.last_hidden_state.shape}')
        cls_vector = outputs.last_hidden_state[:, 0, :]
        print(f'[cls] vector : {cls_vector}')
        logits = self.custom_head(cls_vector)  # shape: (배치크기, num_labels)
        return logits


model_name = "distilbert-base-uncased"
config = AutoConfig.from_pretrained(model_name)
config.num_labels = 2  # 예: 분류 클래스
config.classifier_dropout = 0.3

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = CustomClassifier(config)

sentences = [
    "I really loved this movie, it was fantastic!",
    "This was a waste of time, I hated it.",
]


inputs = tokenizer(sentences, padding=True, truncation=True, max_length=128, return_tensors="pt")
model.eval()

with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

print("모델 출력 (logits):\n", logits)

probs = torch.softmax(logits, dim=-1)
print("\n확률로 변환 (softmax):\n", probs)

predictions = torch.argmax(probs, dim=-1)
print("\n예측 클래스 (0=부정, 1=긍정):\n", predictions)

# 6page - 내용 모델을 local에 저장
save_path = "./my_custom_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)