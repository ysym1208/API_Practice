import sentencepiece as spm

# 입력 파일과 출력 모델 파일 경로 설정
input_file = 'training_text.txt'
model_prefix = 'tokenizer'

# SentencePiece 모델 학습
spm.SentencePieceTrainer.Train(f'--input={input_file} --model_prefix={model_prefix} --vocab_size=30')
