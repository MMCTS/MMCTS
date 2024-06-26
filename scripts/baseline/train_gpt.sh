export CUDA_VISIBLE_DEVICES=1
seed=1

CUDA_VISIBLE_DEVICES=1 accelerate launch --gpu_ids 1 baselines/dialoggpt/train_dialoggpt.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --tokenizer gpt2 \
    --plm_model gpt2 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --num_warmup_steps 6345   \
    --max_sequence_length 512 \
    --max_target_length 80 \
    --learning_rate 5e-5 \
    --output_dir ./gpt2_model/ \
    --seed ${seed}