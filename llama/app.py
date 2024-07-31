import os
import sys
from flask import Flask, request, jsonify
import pandas as pd
import torch
from llama.generation import generate_feedback  # 상대 경로로 임포트

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from tokenizer import Tokenizer
from model import Transformer, ModelArgs

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# 현재 작업 디렉토리 확인 및 파일 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, 'modified_user_performance_with_ids.xlsx')

# 엑셀 파일 읽기
if os.path.exists(file_path):
    df = pd.read_excel(file_path)
    print(f"Loaded initial data from {file_path}")
else:
    df = pd.DataFrame()

tokenizer_path = os.path.join(os.path.dirname(__file__), 'tokenizer.model')
model_weights_path = os.path.join(os.path.dirname(__file__), 'model_weights.pth')

tokenizer = Tokenizer(tokenizer_path)

# 기존 가중치 파일에 맞는 모델 아키텍처
model_args = ModelArgs(
    dim=256,
    n_layers=2,
    n_heads=4,
    vocab_size=30,
    multiple_of=256,
    norm_eps=1e-5,
    max_batch_size=1,
    max_seq_len=128
)

model = Transformer(model_args).to('cpu')  # 모델을 CPU로 이동

if os.path.exists(model_weights_path):
    state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=False)
else:
    print(f"Model weights file not found at {model_weights_path}")

user_data = {}

@app.route('/')
def index():
    return "Welcome to the Interactive API with Excel Data!"

@app.route('/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        print(f"File saved to {file_path}")

        try:
            df = pd.read_excel(file_path)
            print("DataFrame loaded successfully")
        except Exception as e:
            print(f"Failed to load DataFrame: {e}")
            return "Failed to load file", 500

        user_data.clear()

        print("Iterating through DataFrame:")
        for index, row in df.iterrows():
            print(f"Row {index}: {row.to_dict()}")
            user = row['User']
            if user not in user_data:
                user_data[user] = {}
            user_data[user][row['Subject']] = row['Grade']

        print("Uploaded Data:", user_data)  # 업로드된 데이터 확인

        return "File successfully uploaded and data stored", 200

@app.route('/get_grades', methods=['GET'])
def get_grades():
    user = request.args.get('user')
    print(f"Requested grades for user: {user}")
    print(f"Current user_data: {user_data}")
    if user not in user_data:
        return "User not found", 404
    return jsonify(user_data[user])

@app.route('/generate_feedback', methods=['POST'])
def generate_feedback_route():
    data = request.json
    inputs = data['inputs']
    
    # 피드백 생성
    feedback = generate_feedback(inputs, device)
    
    return jsonify({'feedback': feedback})

if __name__ == '__main__':
    app.run(debug=True)












