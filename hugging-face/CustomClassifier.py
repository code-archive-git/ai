import torch.nn as nn
from transformers import AutoModel, PreTrainedModel, AutoConfig


# "Backbone + 커스텀 Head"를 하나의 모델로 묶습니다.
class CustomClassifier(PreTrainedModel):

    config_class = AutoConfig # config.json 의 명시된 class 를 읽도록 설정

    def __init__(self, config):
        print(f'config : {config}')
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
