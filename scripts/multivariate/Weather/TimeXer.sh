export CUDA_VISIBLE_DEVICES=0

model_name=TimeXer


/usr/local/envs/myenv/bin/python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/\
  --data_path DT_0020.csv \
  --model_id DT_0020_96_96 \
  --model $model_name \
  --data TIDE \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
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

/usr/local/envs/myenv/bin/python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ \
  --data_path DT_0020.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data TIDE \
  --features MS \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --factor 3 \
  --enc_in 4 \
  --dec_in 4 \
  --c_out 1 \
  --des 'Exp' \
  --d_model 128 \
  --d_ff 1024 \
  --batch_size 4 \
  --num_workers 8 \
  --itr 1
