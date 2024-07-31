import torch
from model import Transformer, ModelArgs
from tokenizer import Tokenizer

# 실제 토크나이저 모델 파일 경로로 변경
tokenizer_path = '/Users/yunsoyul/Desktop/API_practice/llama/tokenizer.model'
tokenizer = Tokenizer(tokenizer_path)

# 모델 초기화
model_args = ModelArgs(
    dim=256,  # 작은 차원으로 설정
    n_layers=2,  # 레이어 수를 줄입니다.
    n_heads=4,  # 헤드 수를 줄입니다.
    vocab_size=tokenizer.sp_model.vocab_size(),
    max_batch_size=1,
    max_seq_len=64
)
model = Transformer(model_args)

# 모델 가중치 파일 경로 설정
model_weights_path = '/Users/yunsoyul/Desktop/API_practice/llama/model_weights.pth'

# 모델 가중치 저장
torch.save(model.state_dict(), model_weights_path)
print(f"Model weights saved at {model_weights_path}")

