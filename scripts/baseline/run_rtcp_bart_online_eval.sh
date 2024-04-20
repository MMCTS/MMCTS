export CUDA_VISIBLE_DEVICES=5
seed=1111

# CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp_bart.py \
#    --dataset durecdial \
#    --train_data_path data/DuRecDial/data/en_train.txt \
#    --dev_data_path data/DuRecDial/data/en_dev.txt \
#    --test_data_path data/DuRecDial/data/en_test.txt \
#    --policy_tokenizer bert-base-cased \
#    --plm_policy_model bert-base-cased \
#    --generation_tokenizer facebook/bart-base \
#    --plm_generation_model facebook/bart-base \
#    --know_generation_tokenizer facebook/bart-base \
#    --plm_know_generation_model facebook/bart-base \
#    --lm_size 768 \
#    --ffn_size 3072 \
#    --n_layers 12 \
#    --n_heads 8 \
#    --fc_size 128 \
#    --max_sequence_length 512 \
#    --policy_model_path ./rtcp/ \
#    --generation_model_path ./generation_model/ \
#    --know_generation_model_path ./know_generation_model/ \
#    --num_items 100 \
#    --target_set_path ./target_set_full_${seed}/ \
#    --horizon 5 \
#    --use_llm_score \
#    --n 5 \
#    --k 5 \
#    --epsilon 1.0 \
#    --seed ${seed}

CUDA_VISIBLE_DEVICES=5 accelerate launch --gpu_ids 5 baselines/rtcp/online_evaluation_rtcp_bart.py \
    --dataset inspired \
    --train_data_path data/INSPIRED/data/dialog_data/train.tsv \
    --dev_data_path data/INSPIRED/data/dialog_data/dev.tsv \
    --test_data_path data/INSPIRED/data/dialog_data/test.tsv \
    --policy_tokenizer bert-base-cased \
    --plm_policy_model bert-base-cased \
    --generation_tokenizer facebook/bart-base \
    --plm_generation_model facebook/bart-base \
    --know_generation_tokenizer facebook/bart-base \
    --plm_know_generation_model facebook/bart-base \
    --lm_size 768 \
    --ffn_size 3072 \
    --n_layers 12 \
    --n_heads 8 \
    --fc_size 128 \
    --max_sequence_length 512 \
    --policy_model_path ./rtcp_inspired_retrained/ \
    --generation_model_path ./generation_model_inspired/ \
    --know_generation_model_path ./know_generation_model_inspired/ \
    --num_items 55 \
    --target_set_path ./target_set_full_${seed}_inspired/ \
    --horizon 7 \
    --use_llm_score \
    --n 5 \
    --k 5 \
    --epsilon 1.0 \
    --seed ${seed}