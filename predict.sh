#!/bin/bash

echo "Starting predictions for all models..."

# --- 1. DT_0001 모델 예측 ---
echo "--- Predicting for DT_0001 ---"
SETTING_NAME_0001="long_term_forecast_DT_0001_144_72_TimeXer_TIDE_ftMS_sl144_ll96_pl72_dm256_nh8_el1_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"
CHECKPOINT_PATH_0001="./checkpoints/$SETTING_NAME_0001"
INPUT_FILE_0001="./dataset/DT_0001_test_data_with_residual.csv"
EVAL_FILE="./dataset/DT_0001_test_data_with_residual.csv"

# --- ★★★ 아래 predict.py 실행 명령어 전체를 교체하세요 ★★★ ---
/usr/local/envs/myenv/bin/python predict.py \
    --checkpoint_path $CHECKPOINT_PATH_0001/checkpoint.pth \
    --scaler_path $CHECKPOINT_PATH_0001/scaler.gz \
    --input_file $INPUT_FILE_0001 \
    --model TimeXer \
    --task_name long_term_forecast \
    --features MS \
    --seq_len 144 \
    --pred_len 72 \
    --label_len 96 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 1 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 1 \
    --d_layers 1 \
    --d_ff 512 \
    --factor 3 \
    --expand 2 \
    --d_conv 4 \
    --patch_len 16 \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --use_norm 1 \
    --patch_len 16 \
    --freq 't'

/usr/local/envs/myenv/bin/python predict.py \
    --checkpoint_path $CHECKPOINT_PATH_0001/checkpoint.pth \
    --scaler_path $CHECKPOINT_PATH_0001/scaler.gz \
    --evaluate_file $EVAL_FILE \
    --model TimeXer \
    --task_name long_term_forecast \
    --features MS \
    --seq_len 144 \
    --pred_len 72 \
    --label_len 96 \
    --enc_in 5 \
    --dec_in 5 \
    --c_out 1 \
    --d_model 256 \
    --n_heads 8 \
    --e_layers 1 \
    --d_layers 1 \
    --d_ff 512 \
    --factor 3 \
    --expand 2 \
    --d_conv 4 \
    --patch_len 16 \
    --dropout 0.1 \
    --embed timeF \
    --activation gelu \
    --use_norm 1 \
    --patch_len 16 \
    --freq 't'

echo "All predictions are complete."