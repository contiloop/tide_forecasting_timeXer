#!/bin/bash

echo "Starting predictions for all models..."

# --- 1. DT_0001 모델 예측 ---
echo "--- Predicting for DT_0001 ---"
# 학습 시 생성된 setting 폴더 이름을 정확히 맞춰야 합니다.
SETTING_NAME_0001="long_term_forecast_DT_0001_72_72_TimeXer_TIDE_ftMS_sl192_ll96_pl96_dm256_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
CHECKPOINT_PATH_0001="./checkpoints/$SETTING_NAME_0001"
INPUT_FILE_0001="./dataset/DT_0001_test_data_with_residual.csv" # DT_0001에 대한 입력 데이터

/usr/local/envs/myenv/bin/python predict.py \
    --checkpoint_path $CHECKPOINT_PATH_0001/checkpoint.pth \
    --scaler_path $CHECKPOINT_PATH_0001/scaler.gz \
    --input_file $INPUT_FILE_0001 \
    --seq_len 72 \
    --pred_len 72 \
    --enc_in 5 \
    --c_out 1 \
    --features MS \
    --d_model 256 \
    --d_ff 512 \
    --n_heads 8 \
    --e_layers 1

echo "All predictions are complete."