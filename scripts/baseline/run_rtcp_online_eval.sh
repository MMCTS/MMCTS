export CUDA_VISIBLE_DEVICES=5
seed=3

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp.py \
    --train_data_path data/DuRecDial/data/en_train.txt \
    --dev_data_path data/DuRecDial/data/en_dev.txt \
    --test_data_path data/DuRecDial/data/en_test.txt \
    --policy_tokenizer bert-base-cased \
    --policy_plm_model bert-base-cased \
    --generation_tokenizer gpt2 \
    --generation_plm_model gpt2 \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --num_tokens 100 \
    --n_goal_toks 2 \
    --n_topic_toks 2 \
    --use_goal_topic \
    --freeze_plm \
    --max_sequence_length 512 \
    --output_dir ./rtcp/ \
    --num_items 100 \
    --target_set_path ./target_set_${seed}/ \
    --horizon 5 \
    --seed ${seed}
