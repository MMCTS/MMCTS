export CUDA_VISIBLE_DEVICES=5
seed=24

# CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/ppdpp/run.py \
#     --dataset durecdial \
#     --train_data_path data/DuRecDial/data/en_train.txt \
#     --dev_data_path data/DuRecDial/data/en_dev.txt \
#     --test_data_path data/DuRecDial/data/en_test.txt \
#     --tokenizer roberta-large \
#     --plm_model roberta-large \
#     --generation_tokenizer facebook/bart-base \
#     --plm_generation_model facebook/bart-base \
#     --know_generation_tokenizer facebook/bart-base \
#     --plm_know_generation_model facebook/bart-base \
#     --max_train_steps 10 \
#     --sample_times 100 \
#     --save_num 5 \
#     --n 3 \
#     --learning_rate 1e-6 \
#     --lm_size 768 \
#     --max_sequence_length 512 \
#     --policy_model_path ./ppdpp_22/ \
#     --generation_model_path ./generation_model/ \
#     --know_generation_model_path ./know_generation_model/ \
#     --target_set_path ./train_target_set/ \
#     --num_items 100 \
#     --horizon 5 \
#     --epsilon 1.0 \
#     --seed ${seed}


CUDA_VISIBLE_DEVICES=6 accelerate launch --gpu_ids 6 baselines/ppdpp/run.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --tokenizer roberta-large \
    --plm_model roberta-large \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --max_train_steps 10 \
    --sample_times 55 \
    --save_num 5 \
    --n 3 \
    --learning_rate 1e-6 \
    --lm_size 768 \
    --max_sequence_length 512 \
    --policy_model_path ./ppdpp_1_inspired/ \
    --generation_model_path ./generation_model_inspired/ \
    --know_generation_model_path ./know_generation_model_inspired/ \
    --target_set_path ./train_target_set_inspired/ \
    --num_items 100 \
    --horizon 7 \
    --epsilon 1.0 \
    --seed ${seed}