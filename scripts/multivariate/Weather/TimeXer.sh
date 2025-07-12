export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer


/usr/local/envs/myenv/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/\
  --data_path DT_0001.csv \
  --model $model_name \
  --data TIDE \
  --features MS \
  --seq_len 192 \
  --label_len 96 \
  --pred_len 96 \
  --model_id DT_0001_192_96 \
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
  --itr 1

/usr/local/envs/myenv/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/\
  --data_path DT_0020.csv \
  --model $model_name \
  --data TIDE \
  --features MS \
  --seq_len 192 \
  --label_len 96 \
  --pred_len 96 \
  --model_id DT_0020_192_96 \
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
  --itr 1

/usr/local/envs/myenv/bin/python -u run.py \
  --task_name long_term_forecast \
  --is_training 0 \
  --root_path ./dataset/\
  --data_path DT_0057.csv \
  --model $model_name \
  --data TIDE \
  --features MS \
  --seq_len 192 \
  --label_len 96 \
  --pred_len 96 \
  --model_id DT_0057_192_96 \
  --e_layers 1 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 512 \
  --batch_size 4 \
  --num_workers 8 \
  --itr 1
