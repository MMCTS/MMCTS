import itertools
import os
import argparse
import numpy as np

import torch
from transformers import AutoModel, AutoTokenizer, BartForConditionalGeneration
from sentence_transformers import SentenceTransformer

from dyna_gym.models.policy import PolicyModel, load_model
from baselines.rtcp.policy import PolicyModel as RTCPPolicyModel
from dataset.durecdial import DuRecdial
from dataset.inspired import Inspired
from config.config import special_tokens_dict, DURECDIALGOALS
from dataset.data_utils import create_target_set, load_binary_file, save_binary_file

from dyna_gym.envs.utils import reward_func, random_seed
from eval.mcts_eval_online import MCTSCRSOnlineEval
from retrieval.utils import construct_mcts_memory, load_memory_from_file, construct_memory_loaded_from_file
from retrieval.retrieval import Memory


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--output_dir", type=str, help="Where to store the final model.")
    parser.add_argument("--debug", action='store_true', help="Debug mode.")
    # data
    parser.add_argument("--dataset", type=str, default='durecdial', help="A file containing all data.")
    parser.add_argument('--num_items', default=10, type=int, help="max length of both encoder and decoder input.")
    parser.add_argument("--train_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--dev_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--test_data_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument("--memory_path", type=str, required=True, help="A file containing all data.")
    parser.add_argument('--max_sequence_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--max_gen_length', type=int, help="max length of both encoder and decoder input.")
    parser.add_argument('--horizon', type=int, default=5, help="max length of both encoder and decoder input.")
    parser.add_argument('--rollouts', type=int, default=20, help="number of rollout in MCT")
    parser.add_argument('--width', type=int, default=3, help="abc")
    parser.add_argument('--gamma', type=float, default=1., help="abc")

    parser.add_argument('--alg', type=str, default='uct', help="criterion for the selection step")
    parser.add_argument('--policy_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--generation_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--know_generation_model_path', type=str, help="criterion for the selection step")
    parser.add_argument('--target_set_path', type=str, help="criterion for the selection step")
    parser.add_argument("--top_k", default=10, type=int,
                        help="number of retrieval used for computing the memory reward")

    parser.add_argument("--use_llm_score", action="store_true", help="whether to use llm based assessment")
    parser.add_argument("--n", default=5, type=int, help="number of time prompting the llms")
    parser.add_argument("--k", default=3, type=int, help="number of turn used to compute the sr@k")
    parser.add_argument("--epsilon", default=1.0, type=float, help="whether to use llm based assessment")
    parser.add_argument("--use_demonstration", action="store_true", help="whether to use llm based assessment")

    # common
    parser.add_argument("--plm_policy_model", type=str)
    parser.add_argument("--policy_tokenizer", type=str)

    parser.add_argument("--hidden_size", type=int)
    parser.add_argument("--lm_size", type=int)
    parser.add_argument("--use_rtcp_policy", action="store_true", help="whether to use wandb")
    parser.add_argument("--use_vanilla_mcts", action="store_true", help="whether to use wandb")

    # greedy
    parser.add_argument("--plm_generation_model", type=str)
    parser.add_argument("--generation_tokenizer", type=str)
    parser.add_argument("--plm_know_generation_model", type=str)
    parser.add_argument("--know_generation_tokenizer", type=str)
    parser.add_argument("--offline_policy", action="store_true", help="whether to use offline policy")
    parser.add_argument("--use_training_data", action="store_true", help="whether to use offline policy")

    # rtcp policy
    parser.add_argument("--ffn_size", type=int, default=128)
    parser.add_argument("--fc_size", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=8)

    # text generation
    parser.add_argument("--use_llama2", action="store_true", help="whether to use offline policy")

    # wandb
    parser.add_argument("--use_wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--entity", type=str, help="wandb username")
    parser.add_argument("--project", type=str, help="wandb exp project")
    parser.add_argument("--name", type=str, help="wandb exp name")
    parser.add_argument("--log_all", action="store_true", help="log in all processes, otherwise only in rank0")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse argments
    args = parse_args()

    random_seed(args.seed)

    device = torch.device('cuda:0')
    # arguments for the UCT agent
    uct_args = dict(
        rollouts=args.rollouts,
        gamma=args.gamma,
        width=args.width,
        alg=args.alg,  # or p_uct
        k=args.top_k  # num retrieval
    )

    # will be passed to huggingface model.generate()
    model_generation_args = dict()

    plm_policy_model = args.plm_policy_model
    policy_model_path = args.policy_model_path
    policy_model_name = 'policy.pth'
    lm_size = args.lm_size
    hidden_size = args.hidden_size

    if args.dataset == 'durecdial':
        dataset = DuRecdial(
            train_data_path=args.train_data_path,
            dev_data_path=args.dev_data_path,
            test_data_path=args.test_data_path
        )
    elif args.dataset == 'inspired':
        dataset = Inspired(
            train_data_path=args.train_data_path,
            dev_data_path=args.dev_data_path,
            test_data_path=args.test_data_path
        )

    # goal2id = {k: v for v, k in enumerate(DURECDIALGOALS)}

    # use greedy policy as the default policy
    if not args.use_rtcp_policy:
        ori_goal2id = None
        topic2id = None
        goal2id = load_binary_file(os.path.join(policy_model_path, "goal2id.pkl"))
        # create and load the weights for policy model

        policy_plm = AutoModel.from_pretrained(plm_policy_model)
        policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer)
        policy_tokenizer.add_special_tokens(special_tokens_dict)
        policy_plm.resize_token_embeddings(len(policy_tokenizer))

        policy_model = PolicyModel(
            plm=policy_plm,
            n_goals=len(goal2id),
            hidden_size=args.hidden_size,
            lm_size=args.lm_size
        )
    # if we use RTCP as the default policy
    else:
        ori_goal2id = load_binary_file(os.path.join(policy_model_path, 'rtcp_goal2id.pkl'))
        topic2id = load_binary_file(os.path.join(policy_model_path, 'rtcp_topic2id.pkl'))

        id2goal = {v: k for k, v in ori_goal2id.items()}
        id2topic = {v: k for k, v in topic2id.items()}

        all_goals = []
        all_topics = []

        # loop overall goal
        for id in range(len(ori_goal2id.keys())):
            goal = id2goal[id]
            all_goals.append(goal)

        # loop overall topic
        for id in range(len(topic2id.keys())):
            topic = id2topic[id]
            all_topics.append(topic)

        # combine goal and topic
        goal2id = list(itertools.product(all_goals, all_topics))
        goal2id = {k: v for v, k in enumerate(goal2id)}

        # switch from predicting a goal to predicting a pair of a goal and a topic
        # goal2id = itertools.product(dataset.goals, dataset.topics)
        # goal2id = {k: v for v, k in enumerate(goal2id)}

        context_encoder = AutoModel.from_pretrained(args.plm_policy_model)
        path_encoder = AutoModel.from_pretrained(args.plm_policy_model)
        policy_tokenizer = AutoTokenizer.from_pretrained(args.policy_tokenizer)

        policy_tokenizer.add_special_tokens(special_tokens_dict)
        context_encoder.resize_token_embeddings(len(policy_tokenizer))
        path_encoder.resize_token_embeddings(len(policy_tokenizer))

        policy_model = RTCPPolicyModel(
            context_encoder=context_encoder,
            path_encoder=path_encoder,
            knowledge_encoder=None,  # do not have knowledge,
            n_layers=args.n_layers,
            n_topics=len(dataset.topics),
            n_goals=len(dataset.goals),
            n_heads=args.n_heads,
            lm_hidden_size=args.lm_size,
            ffn_size=args.ffn_size,
            fc_hidden_size=args.fc_size
        )

    policy_model = load_model(policy_model, os.path.join(policy_model_path, policy_model_name))
    policy_model.to(device)

    # create and load the weights for knowledge generation model
    plm_know_generation_model = args.plm_know_generation_model
    know_generation_model_path = args.know_generation_model_path
    know_generation_model_name = 'know_generation.pth'
    know_generation_model = BartForConditionalGeneration.from_pretrained(plm_know_generation_model)

    know_generation_tokenizer = AutoTokenizer.from_pretrained(args.know_generation_tokenizer)
    know_generation_tokenizer.add_special_tokens(special_tokens_dict)
    know_generation_model.resize_token_embeddings(len(know_generation_tokenizer))
    know_generation_model = load_model(know_generation_model,
                                       os.path.join(know_generation_model_path, know_generation_model_name))
    know_generation_model.to(device)

    # create and load the weights for generation model
    plm_generation_model = args.plm_generation_model
    generation_model_path = args.generation_model_path
    generation_model_name = 'response_generation.pth'
    generation_model = BartForConditionalGeneration.from_pretrained(plm_generation_model)

    generation_tokenizer = AutoTokenizer.from_pretrained(args.generation_tokenizer)
    generation_tokenizer.add_special_tokens(special_tokens_dict)
    generation_model.resize_token_embeddings(len(generation_tokenizer))
    generation_model = load_model(generation_model, os.path.join(generation_model_path, generation_model_name))
    generation_model.to(device)

    if not os.path.exists(args.target_set_path):
        os.mkdir(args.target_set_path)

    if os.path.exists(os.path.join(args.target_set_path, "target.pkl")):
        target_set = load_binary_file(os.path.join(args.target_set_path, "target.pkl"))
    else:
        # create the target item set.
        target_set = create_target_set(dataset.train_convs, dataset.test_instances, num_items=args.num_items)
        save_binary_file(target_set, os.path.join(args.target_set_path, "target.pkl"))

    # retrieval model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # build memory for mcts using the training dataset.
    if args.use_training_data:
        raw_memory = construct_mcts_memory(dataset.train_instances, target_set)
        memory = Memory(
            embedding_model=embedding_model,
            raw_memory=raw_memory,
            instances=dataset.train_instances,
            scores=None
        )
    else:
        raw_memory = load_memory_from_file(args.memory_path)
        raw_states, raw_continuations, raw_scores = construct_memory_loaded_from_file(raw_memory)
        memory = Memory(
            embedding_model=embedding_model,
            raw_memory=raw_states,
            instances=raw_continuations,
            scores=raw_scores,
            d_model=384
        )

    memory.train_convs = dataset.train_convs

    # None if we use vanilla MCTS.
    if args.use_vanilla_mcts:
        memory = None

    terminal_act = "Say goodbye"
    mcts_online_eval = MCTSCRSOnlineEval(
        target_set=target_set,
        terminal_act=terminal_act,
        use_llm_score=args.use_llm_score,
        epsilon=args.epsilon,
        n=args.n,
        k=args.k,
        use_demonstration=args.use_demonstration,
        generation_model=generation_model,
        generation_tokenizer=generation_tokenizer,
        know_generation_model=know_generation_model,
        know_generation_tokenizer=know_generation_tokenizer,
        policy_model=policy_model,
        policy_tokenizer=policy_tokenizer,
        memory=memory,
        horizon=args.horizon,
        reward_func=reward_func,
        uct_args=uct_args,
        goal2id=goal2id,
        device=device,
        offline_policy=args.offline_policy,
        max_sequence_length=args.max_sequence_length,
        max_gen_length=args.max_gen_length,
        model_generation_args=model_generation_args,
        should_plot_tree=True,  # plot the tree after generation,
        use_rtcp_policy=args.use_rtcp_policy,  # if use rtcp as the policy
        use_llama2=args.use_llama2,  # if use llama2 as text generation model
        dataset=args.dataset,  # dataset
        topic2id=[ori_goal2id, topic2id]  # only work for rtcp policy
    )

    model_name = "offline" if args.offline_policy else "mcts"

    # policy model / target_set_id / mcts (offline)/ generated_conversations.txt
    saved_file_path = os.path.join(args.policy_model_path, f"target_set_sub_sampled_{args.seed}")
    if not os.path.exists(saved_file_path):
        os.mkdir(saved_file_path)

    saved_file_path = os.path.join(saved_file_path, model_name)
    if not os.path.exists(saved_file_path):
        os.mkdir(saved_file_path)

    saved_convs_path = os.path.join(saved_file_path, "generated_conversations.txt")
    save_experience_path = os.path.join(saved_file_path, "generated_experience.txt")

    # no experience for greedy policy
    if args.offline_policy:
        save_experience_path = None

    # compute online evaluation metrics
    srk, sr, avg_turn, sr_turns, o_sr, _, domain_specific_metrics = mcts_online_eval.eval(saved_convs_path,
                                                                                          save_experience_path)

    print(f"Success rate @ {args.k}:", srk)
    print("Success rate:", sr)
    print("Avg turn: ", avg_turn)
    print("Success rate turns: ", sr_turns)

    print("Objective Success rate:", o_sr)

    print("Domain specific metrics: ", domain_specific_metrics)

    print("State value reward variance: ", np.var(mcts_online_eval.global_reward_his))
