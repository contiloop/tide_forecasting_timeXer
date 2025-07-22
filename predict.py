import torch
import numpy as np
import pandas as pd
import argparse
import joblib
import os
from models import TimeXer # 사용하는 모델에 맞게 수정

# 1. 인자 파싱 (필요한 정보만)
parser = argparse.ArgumentParser(description='Time Series Prediction')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file (.pth)')
parser.add_argument('--scaler_path', type=str, required=True, help='Path to the saved scaler file (.gz)')
parser.add_argument('--input_file', type=str, required=True, help='Path to the new input data CSV file')
# 모델 구조를 다시 만들기 위해 필요한 하이퍼파라미터들
parser.add_argument('--seq_len', type=int, required=True, help='input sequence length')
parser.add_argument('--pred_len', type=int, required=True, help='prediction sequence length')
parser.add_argument('--enc_in', type=int, required=True, help='encoder input size')
parser.add_argument('--c_out', type=int, required=True, help='output size')
parser.add_argument('--d_model', type=int, required=True, help='dimension of model')
parser.add_argument('--n_heads', type=int, required=True, help='num of heads')
parser.add_argument('--e_layers', type=int, required=True, help='num of encoder layers')
parser.add_argument('--d_ff', type=int, required=True, help='dimension of fcn')
parser.add_argument('--features', type=str, required=True, help='M, S, or MS')
# TimeXer 모델을 위한 추가 인자들
parser.add_argument('--expand', type=int, default=2)
parser.add_argument('--d_conv', type=int, default=4)
parser.add_argument('--factor', type=int, default=3)
parser.add_argument('--label_len', type=int, default=96)
parser.add_argument('--dec_in', type=int, default=5)
parser.add_argument('--d_layers', type=int, default=1)

args = parser.parse_args()

# 2. 예측 함수
def predict(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 모델 구조 초기화 및 Weight 로드
    model = TimeXer.Model(args).float().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # 스케일러 로드
    scaler = joblib.load(args.scaler_path)
    print("Scaler loaded successfully.")

    # 새로운 입력 데이터 로드
    df_input = pd.read_csv(args.input_file)
    if 'date' in df_input.columns:
        df_input = df_input.drop(columns=['date'])
    raw_input = df_input.tail(args.seq_len).values
    
    if raw_input.shape[0] < args.seq_len:
        raise ValueError(f"Input data must have at least {args.seq_len} time steps.")
    if raw_input.shape[1] != args.enc_in:
        raise ValueError(f"Input data features should be {args.enc_in} but got {raw_input.shape[1]}")

    # 데이터 스케일링
    input_scaled = scaler.transform(raw_input)
    batch_x = torch.from_numpy(input_scaled).float().unsqueeze(0).to(device)
    
    # 예측 실행
    with torch.no_grad():
        outputs = model(batch_x, None, None, None)

    # 결과 후처리 및 스케일 복원
    prediction_scaled = outputs.detach().cpu().numpy()[0]
    if args.features == 'MS':
        padding = np.zeros((prediction_scaled.shape[0], scaler.n_features_in_ - 1))
        prediction_padded = np.concatenate((padding, prediction_scaled), axis=1)
        prediction = scaler.inverse_transform(prediction_padded)[:, -1]
    else:
        prediction = scaler.inverse_transform(prediction_scaled)
        
    return prediction

# 3. 메인 로직
if __name__ == '__main__':
    final_prediction = predict(args)
    print("\n--- Prediction Result ---")
    print(final_prediction.flatten())
    
    np.save('prediction_result.npy', final_prediction)
    print("\n✅ Prediction saved to 'prediction_result.npy'")