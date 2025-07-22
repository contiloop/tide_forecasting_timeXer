export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer


/usr/local/envs/myenv/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/\
  --data_path DT_0001.csv \
  --model $model_name \
  --target residual \
  --data TIDE \
  --features MS \
  --seq_len 72 \
  --label_len 96 \
  --pred_len 72 \
  --model_id DT_0001_72_72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --num_workers 8 \
  --output_attention \
  --itr 1

/usr/local/envs/myenv/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/\
  --data_path DT_0001.csv \
  --model $model_name \
  --target residual \
  --data TIDE \
  --features MS \
  --seq_len 144 \
  --label_len 96 \
  --pred_len 72 \
  --model_id DT_0001_144_72 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 5 \
  --dec_in 5 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --num_workers 8 \
  --output_attention \
  --itr 1

