#!/bin/bash

echo "Starting All Inference Tasks for DT_0001..."

# --- 공통 설정 ---
SETTING_NAME="long_term_forecast_DT_0001_144_72_TimeXer_TIDE_ftMS_sl144_ll96_pl72_dm256_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
CHECKPOINT_PATH="./checkpoints/$SETTING_NAME"
DATA_FILE="./dataset/DT_0001_test_data_with_residual.csv"
PYTHON_EXEC="/usr/local/envs/myenv/bin/python"
# ----------------

# --- 1. 단일 미래 예측 실행 ---
echo -e "\n--- Running Single Future Prediction ---"
$PYTHON_EXEC inference.py \
    --checkpoint_path $CHECKPOINT_PATH/checkpoint.pth \
    --scaler_path $CHECKPOINT_PATH/scaler.gz \
    --predict_input_file $DATA_FILE \
    --model TimeXer \
    --features MS \
    --seq_len 144 \
    --pred_len 72 \
    --label_len 96 \
    --enc_in 5 \
    --c_out 1 \
    --d_model 256 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --patch_len 16 \
    --dec_in 5 \
    --expand 2 \
    --d_conv 4


# --- 2. 전체 기간 성능 평가 실행 ---
echo -e "\n--- Running Rolling Evaluation ---"
$PYTHON_EXEC inference.py \
    --checkpoint_path $CHECKPOINT_PATH/checkpoint.pth \
    --scaler_path $CHECKPOINT_PATH/scaler.gz \
    --evaluate_file $DATA_FILE \
    --model TimeXer \
    --features MS \
    --seq_len 144 \
    --pred_len 72 \
    --label_len 96 \
    --enc_in 5 \
    --c_out 1 \
    --d_model 256 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 1 \
    --d_layers 1 \
    --factor 3 \
    --patch_len 16 \
    --dec_in 5 \
    --expand 2 \
    --d_conv 4


echo -e "\nAll tasks are complete."