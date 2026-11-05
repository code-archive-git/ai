import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, PreTrainedModel, AutoConfig

print(f'GPU 사용가능 여부 : {torch.cuda.is_available()}')
print(torch.__version__)            # 뒤에 +cpu 붙어있으면 CPU 전용 빌드


# -----------------------------------------------------------------------
# 1단계: 커스텀 모델 클래스 만들기
# -----------------------------------------------------------------------
# "Backbone + 커스텀 Head"를 하나의 모델로 묶습니다.
class CustomClassifier(PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        # (1) Backbone 불러오기
        #     AutoModel 은 "헤드가 없는" 순수 backbone만 불러옵니다.
        #     (AutoModelForSequenceClassification과 다르게, 이미 만들어진
        #      분류 헤드가 붙어있지 않습니다 -> 우리가 직접 헤드를 붙일 것!)
        self.backbone = AutoModel.from_config(config)
        num_labels = getattr(config, "num_labels", 2)  # config에 설정된 num_labels 사용 (기본값 2)
        dropout_prob = getattr(config, "classifier_dropout", 0.3)  # 드롭아웃 비율

        # backbone이 출력하는 벡터의 크기(차원)를 알아냅니다.
        # 예: distilbert는 보통 768차원 벡터를 출력합니다.
        hidden_size = config.hidden_size

        # (2) 커스텀 헤드 만들기
        #     여기서는 "선형층 -> 활성화 함수(ReLU) -> 드롭아웃 -> 선형층"
        #     이렇게 2개의 층을 쌓은, 조금 더 표현력 있는 헤드를 만들어봅니다.
        #     (기본 제공 모델은 보통 선형층 1개짜리 헤드만 있습니다)
        self.custom_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # 768 -> 384로 차원 축소
            nn.ReLU(),  # 비선형성 추가
            nn.Dropout(dropout_prob),  # 과적합 방지(무작위 30% 끄기)
            nn.Linear(hidden_size // 2, num_labels),  # 384 -> 2 (최종 클래스 개수)
        )

        # 가중치 초기화 (PreTrainedModel 제공 함수)
        self.post_init()

    # model() 을 쓰는 순간 내부적으로 호출됨
    def forward(self, input_ids, attention_mask=None):
        """
        forward()는 "입력이 들어왔을 때 모델이 어떻게 계산할지"를 정의하는 함수입니다.
        """
        # (1) Backbone에 입력을 넣어서 결과를 받습니다.
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # outputs.last_hidden_state의 shape: (문장, 토큰, 벡터들)
        # [cls] 는 해당 문장의 대표 내용을 담는다.
        # 기존의 언어 생성이나 분석와 같이 단어와 단어의 의미가 필요하지만
        # classification 같은 분류는 이 문장 자체의 의미가 필요하다.
        # 그래서 첫 토큰인 [CLS] 의 모든 벡터를 들고오는 것!
        print(f'shape : {outputs.last_hidden_state.shape}')
        cls_vector = outputs.last_hidden_state[:, 0, :]

        print(f'[cls] vector : {cls_vector}')
        # (2) 우리가 만든 커스텀 헤드에 통과시켜 최종 예측값(logits)을 얻습니다.
        logits = self.custom_head(cls_vector)  # shape: (배치크기, num_labels)
        return logits


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