import torch
import numpy as np
import pandas as pd
import argparse
import joblib
import os
from models import TimeXer # 사용하는 모델에 맞게 수정
import tqdm

# 1. 인자 파싱 (필요한 정보만)
# --- 1. 설정 및 인자 파싱 (이 부분을 전체 교체) ---
parser = argparse.ArgumentParser(description='Time Series Prediction')

# --- 필수 인자 ---
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint file (.pth)')
parser.add_argument('--scaler_path', type=str, required=True, help='Path to the saved scaler file (.gz)')
parser.add_argument('--input_file', type=str, required=True, help='Path to the new input data CSV file')

# --- 모델 아키텍처 인자 (학습 때와 동일하게) ---
parser.add_argument('--model', type=str, default='TimeXer', help='model name') # 모델 이름 추가
parser.add_argument('--task_name', type=str, default='long_term_forecast', help='task name')
parser.add_argument('--seq_len', type=int, required=True, help='input sequence length')
parser.add_argument('--pred_len', type=int, required=True, help='prediction sequence length')
parser.add_argument('--label_len', type=int, required=True, help='start token length')
parser.add_argument('--features', type=str, required=True, help='M, S, or MS')
parser.add_argument('--enc_in', type=int, required=True, help='encoder input size')
parser.add_argument('--dec_in', type=int, required=True, help='decoder input size')
parser.add_argument('--c_out', type=int, required=True, help='output size')
parser.add_argument('--d_model', type=int, required=True, help='dimension of model')
parser.add_argument('--n_heads', type=int, required=True, help='num of heads')
parser.add_argument('--e_layers', type=int, required=True, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, required=True, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, required=True, help='dimension of fcn')
parser.add_argument('--factor', type=int, required=True, help='attn factor')
parser.add_argument('--patch_len', type=int, required=True, help='patch length for TimeXer')
parser.add_argument('--expand', type=int, required=True)
parser.add_argument('--d_conv', type=int, required=True)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF', help='time features encoding')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize')
parser.add_argument('--freq', type=str, default='t', help='freq for time features encoding')

args = parser.parse_args()

# --- 2. 공통 함수: 모델 및 스케일러 로드 ---
def load_model_and_scaler(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TimeXer.Model(args).float().to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    scaler = joblib.load(args.scaler_path)
    print(f"Using device: {device}")
    print("Model and scaler loaded successfully.")
    return model, scaler, device

# --- 3. 모드 1: 단일 미래 예측 함수 ---
def predict_future(args, model, scaler, device):
    df_input = pd.read_csv(args.predict_input_file)
    if 'date' in df_input.columns:
        df_input = df_input.drop(columns=['date'])
    raw_input = df_input.tail(args.seq_len).values
    
    input_scaled = scaler.transform(raw_input)
    batch_x = torch.from_numpy(input_scaled).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(batch_x, None, None, None)[0]
        
    prediction_scaled = outputs.detach().cpu().numpy()[0]
    if args.features == 'MS':
        padding = np.zeros((prediction_scaled.shape[0], scaler.n_features_in_ - 1))
        prediction_padded = np.concatenate((padding, prediction_scaled), axis=1)
        prediction = scaler.inverse_transform(prediction_padded)[:, -1]
    else:
        prediction = scaler.inverse_transform(prediction_scaled)
    return prediction

# --- 4. 모드 2: 전체 기간 롤링 평가 함수 ---
def evaluate_performance(args, model, scaler, device):
    df_eval = pd.read_csv(args.evaluate_file)
    if 'date' in df_eval.columns:
        df_eval = df_eval.drop(columns=['date'])
    raw_data = df_eval.values
    data_scaled = scaler.transform(raw_data)

    preds, trues = [], []
    num_samples = len(data_scaled) - args.seq_len - args.pred_len + 1
    for i in tqdm(range(num_samples), desc="Evaluating"):
        s_begin = i
        s_end = s_begin + args.seq_len
        input_scaled = data_scaled[s_begin:s_end]
        batch_x = torch.from_numpy(input_scaled).float().unsqueeze(0).to(device)

        true_begin = s_end
        true_end = true_begin + args.pred_len
        true_scaled = data_scaled[true_begin:true_end]
        
        with torch.no_grad():
            outputs = model(batch_x, None, None, None)[0]
        
        preds.append(outputs.detach().cpu().numpy()[0])
        trues.append(true_scaled[-args.pred_len:, -1:])

    return np.array(preds), np.array(trues)


# --- 5. 메인 로직 ---
if __name__ == '__main__':
    # 결과 저장 폴더 생성
    output_dir = 'pred_results'
    os.makedirs(output_dir, exist_ok=True)
    
    model, scaler, device = load_model_and_scaler(args)

    if args.predict_input_file:
        print("\n--- Running in Single Prediction Mode ---")
        prediction = predict_future(args, model, scaler, device)
        output_path = os.path.join(output_dir, 'prediction_future.npy')
        np.save(output_path, prediction)
        print(f"\n✅ Future prediction saved to {output_path}")

    elif args.evaluate_file:
        print("\n--- Running in Rolling Evaluation Mode ---")
        eval_preds, eval_trues = evaluate_performance(args, model, scaler, device)
        pred_path = os.path.join(output_dir, 'evaluation_preds.npy')
        true_path = os.path.join(output_dir, 'evaluation_trues.npy')
        np.save(pred_path, eval_preds)
        np.save(true_path, eval_trues)
        print(f"\n✅ Evaluation results saved to {output_dir}")
        print(f"   - Predictions shape: {eval_preds.shape}")
        print(f"   - Truths shape: {eval_trues.shape}")
        
    else:
        print("오류: --predict_input_file 또는 --evaluate_file 중 하나의 모드를 선택해야 합니다.")
