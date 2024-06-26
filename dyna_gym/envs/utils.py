import random
from collections import defaultdict
import copy
import math
import time

import openai
import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import replicate
from tqdm import tqdm

from dataset.data_utils import convert_list_to_str, convert_dict_to_str, convert_example_to_feature_for_goal_prediction, \
    convert_example_to_feature_for_response_generation, convert_example_to_feature_for_knowledge_generation

from baselines.rtcp.utils import predict_action_rtcp
from retrieval.utils import concatenate_sentences

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff


@retry(
    retry=retry_if_exception_type((openai.error.APIError, openai.error.APIConnectionError, openai.error.RateLimitError,
                                   openai.error.ServiceUnavailableError, openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


API_KEY = ""
MODEL = "gpt-3.5-turbo"
openai.api_key = API_KEY
IGNORE_INDEX = -100


# correct solution:
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


def compute_run_time(func, input):
    """
    function that compute the computational time of a given input function
    @param func: the given input function
    @param input: the corresponding input of the given function
    @return: the result of the given function
    """
    t = time.time()
    result = func(**input)
    # print(f"[Function Name]: {func.__name__}, [Run time]: ", time.time() - t)
    return result


def reformat_demonstration(demonstration, is_agent_start=False):
    """
    function that reformat the demonstrative conversation
    @param demonstration: the given conversation
    @param is_agent_start: True if the system starts the conversation else False
    @return: the reformated demonstrative conversation
    """
    new_demonstration = []
    role = 0
    if is_agent_start:
        role = -1
    for utt in demonstration:
        if role % 2 == 0:
            new_demonstration.append({'role': 'user', 'content': utt})
        elif role == -1 or role % 2 != 0:
            new_demonstration.append({'role': 'assistant', 'content': utt})
        role += 1
    return new_demonstration


def generate_sys_resp(state, action):
    """ Generate a system response using ChatGPT.
    Args:
        state (_type_): the current state which consists of task background, pre_topics and prev_goals
        action (_type_): a chosen goal.
    """
    demonstration = state['demonstration']['conversation']
    knowledge_str = convert_list_to_str(state['knowledge'])
    user_profile_str = convert_dict_to_str(state['task_background']['user_profile'])
    target_topic = state['task_background']['target_topic']
    system_instruction_1 = f"""You are a recommender. You will be given a set of relevant knowledge 
        deliminated by triple backticks ```{knowledge_str}``` and information about the user
        deliminated by the following triple backticks ```{user_profile_str}```. Your task is to generate 
        a response following the action ```{action}``` using the given knowledge and user profile. If the action 
        is recommendation, then you need recommend the item {target_topic} to the user. 
        The following is an example conversation between a recommender and an user.
    """.replace('\n', '')
    # the first instruction prompt
    messages = [
        {"role": "system", "content": system_instruction_1},
    ]
    # 1-shot demonstration
    for utt in reformat_demonstration(demonstration,
                                      is_agent_start=state['demonstration']['goal_type_list'][0] == 'Greetings'):
        messages.append(utt)

    system_instruction_2 = """
    The following is a new conversation between a recommender (you) and an user.
    """
    # the second instruction prompt
    messages.append(
        {"role": "user", "content": system_instruction_2},
    )
    # current conversation
    for utt in state['dialogue_context']:
        messages.append(utt)

    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0,
        max_tokens=50
    )
    return response.choices[0]['message']['content']


def get_user_resp(state, sys_response, dataset='inspired'):
    demonstration = state['demonstration']['conversation']
    target_topic = state['task_background']['target_topic']

    # 1-shot demonstration
    if dataset != 'inspired':
        seeker_instruction_0 = ''' This is an example of a {} conversation between the user (you) and the system.
        '''.format(state['demonstration']['target_goal'])
    else:
        seeker_instruction_0 = ''' This is an example of a movie recommendation conversation between the user (you) and the system.
        '''.format(state['demonstration']['target_goal'])

    messages = [
        {"role": "system", "content": seeker_instruction_0},
    ]
    target_goal = state['task_background']['target_goal']
    for utt in reformat_demonstration(demonstration,
                                      is_agent_start=state['demonstration']['goal_type_list'][0] == 'Greetings'):
        # switch role
        if utt['role'] == 'user':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        messages.append(utt)

    if dataset != 'inspired':
        seeker_instruction_2 = '''Now enter the role-playing mode. In the following conversation, 
        you will play as an use. You are the user who is looking for a {}. 
        Please reply with only one short and succinct sentence.
        '''.format(target_goal)
    else:
        seeker_instruction_2 = '''Now enter the role-playing mode. In the following conversation, 
        you will play as an use. You are the user who is looking for a {}. 
        Please reply with only one short and succinct sentence.
        '''.format("movie recommendation", target_topic)

    # the first instruction prompt
    # messages.append({"role": "system", "content": seeker_instruction_1})
    messages.append({"role": "system", "content": seeker_instruction_2})
    new_state = copy.deepcopy(state)
    # current conversation
    for utt in new_state['dialogue_context']:
        # switch role
        if utt['role'] == 'user':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        messages.append(utt)

    # the new generate response.
    messages.append(
        {'role': 'user', 'content': sys_response}
    )

    # getting the response.
    # response = openai.ChatCompletion.create(
    #     model=MODEL,
    #     messages=messages,
    #     temperature=0.0,
    #     max_tokens=50
    # )

    response = chat_completion_with_backoff(
        model=MODEL,
        messages=messages,
        temperature=0.0,
        max_tokens=50
    )

    # print(response)

    return response.choices[0]['message']['content']


def update_state(state, action, sys_response, user_response):
    """function that updates the state of the conversation
    in order to update the state, we need to simulate an user's response using ChatGPT.
    Args:
        state (_type_): the current state of the conversation.
        action (_type_): the chosen goal.
        response (_type_): the generated user response.
    """
    # update state
    new_state = copy.deepcopy(state)
    new_state['dialogue_context'].append(
        {"role": "assistant", "content": sys_response}
    )
    new_state['dialogue_context'].append(
        {"role": "user", "content": user_response}
    )
    if isinstance(action, tuple):
        goal, topic = action
        new_state['pre_goals'].append(goal)
        new_state['pre_topics'].append(topic)
    else:
        new_state['pre_goals'].append(action)
    return new_state


def check_terminated_condition(action, terminated_action):
    """
    function check if the target item appears in the system response.
    @param action: the predicted action by the system
    @param terminated_action: the predefined terminated action
    @return: True if the target appear in the generated response else False
    """
    return action == terminated_action


def generate_knowledge_with_plm(generation_model, tokenizer, action, state, max_sequence_length, max_gen_length=50,
                                pad_to_multiple_of=True, padding='max_length', device=None):
    """
    function that generates a knowledge utterance with a finetuned pretrained language model
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param action: the predicted action
    @param state:  the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param pad_to_multiple_of: True if we pad to multiple instances.
    @param padding: type of padding default = 'max length"
    @param device: device to allocate tensors
    @return: a generated knowledge utterance.
    """
    # convert state to input feature
    input_features = defaultdict(list)

    # assign the predicted goal to the input state
    state['pred_goal'] = action[0]
    # assign the predicted topic to the input state
    state['pred_topic'] = action[1]

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_knowledge_generation(tokenizer=tokenizer, instance=state,
                                                                       max_sequence_length=max_sequence_length,
                                                                       is_test=True)
    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # forward the input features through the model
    gen_seqs = generation_model.generate(
        **input_features,
        max_new_tokens=max_gen_length,
        no_repeat_ngram_size=3
    )
    # remove special tokens
    gen_resp_ids = []
    for gen_seq in gen_seqs:
        gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]


def generate_sys_response_with_plm(generation_model, tokenizer, action, knowledge, state, max_sequence_length,
                                   max_gen_length=50,
                                   pad_to_multiple_of=True, padding='max_length', device=None, dataset='durecdial'):
    """
    function that generates a system response with a finetuned pretrained language model
    @param generation_model: the finetuned huggingface pretrained PLM
    @param tokenizer: a huggingface tokenizer.
    @param action: the predicted action
    @param knowledge: the generated knowledge
    @param state:  the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param pad_to_multiple_of: True if we pad to multiple instances.
    @param padding: type of padding default = 'max length"
    @param device: device to allocate tensors
    @return: a generated system response
    """
    # convert state to input feature
    input_features = defaultdict(list)

    # assign the predicted action to the input state
    state['pred_goal'] = action[0]
    state['pred_topic'] = action[1]
    state['pred_know'] = knowledge

    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_response_generation(tokenizer=tokenizer, instance=state,
                                                                      max_sequence_length=max_sequence_length,
                                                                      is_test=True, dataset=dataset)
    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # forward the input features through the model
    gen_seqs = generation_model.generate(
        **input_features,
        max_new_tokens=max_gen_length,
        no_repeat_ngram_size=3
    )
    # remove special tokens
    gen_resp_ids = []
    for gen_seq in gen_seqs:
        gen_seq = [token_id for token_id in gen_seq if token_id != tokenizer.pad_token_id]
        gen_resp_ids.append(gen_seq)

    decoded_preds = tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=False)
    decoded_preds = [decoded_pred.replace('<pad>', '').replace('<s>', '').replace('</s>', '') for decoded_pred in
                     decoded_preds]
    decoded_preds = [pred.strip() for pred in decoded_preds]

    return decoded_preds[0]


def predict_action(policy_model, tokenizer, state, max_sequence_length, goal2id=None, pad_to_multiple_of=True,
                   padding='max_length', device=None):
    """
    function that predicts an action given the input state
    @param policy_model: the offline policy model
    @param tokenizer: a huggingface tokenizer.
    @param state: the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence
    @param goal2id: a dictionary that map goals to indices
    @param pad_to_multiple_of: pad to multiple instances
    @param padding: type of padding
    @param device: device to allocate tensors
    @return: a predicted action
    """
    input_features = defaultdict(list)
    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_goal_prediction(tokenizer, state, max_sequence_length,
                                                                  goal2id)

    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # compute policy with offline policy model.
    logits = policy_model(input_features)
    pred_action_id = logits.argmax(-1).detach().cpu().numpy().tolist()[0]
    id2goal = {v: k for k, v in goal2id.items()}
    action = id2goal[pred_action_id]
    return action


def predict_action_prior(policy_model, tokenizer, state, max_sequence_length, goal2id=None, pad_to_multiple_of=True,
                         padding='max_length', device=None):
    """
    function that predicts an action given the input state
    @param policy_model: the offline policy model
    @param tokenizer: a huggingface tokenizer.
    @param state: the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence
    @param goal2id: a dictionary that map goals to indices
    @param pad_to_multiple_of: pad to multiple instances
    @param padding: type of padding
    @param device: device to allocate tensors
    @return: a predicted action
    """
    input_features = defaultdict(list)
    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_goal_prediction(tokenizer, state, max_sequence_length,
                                                                  goal2id)

    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # compute policy with offline policy model.
    logits = policy_model(input_features)

    # compute probability distribution
    all_probs = torch.softmax(logits, dim=-1)

    # convert to np.array
    all_probs = all_probs.detach().cpu().numpy()
    return all_probs[0]


def predict_topk_action(policy_model, tokenizer, state, max_sequence_length, goal2id=None, pad_to_multiple_of=True,
                        padding='max_length', device=None, top_k=3, epsilon=0.1):
    """
    function that predicts an action given the input state
    @param policy_model: the offline policy model
    @param tokenizer: a huggingface tokenizer.
    @param state: the current state of the env
    @param max_sequence_length: the maximum number of tokens in the input sequence
    @param goal2id: a dictionary that map goals to indices
    @param pad_to_multiple_of: pad to multiple instances
    @param padding: type of padding
    @param device: device to allocate tensors
    @param top_k: number of sampled actions
    @param epsilon: a small probability used for exploration.
    @return: a predicted action
    """
    input_features = defaultdict(list)
    # convert state to input features
    input_ids, _ = convert_example_to_feature_for_goal_prediction(tokenizer, state, max_sequence_length,
                                                                  goal2id)

    input_features['input_ids'] = input_ids
    # padding the input features
    input_features = tokenizer.pad(
        input_features, padding=padding, pad_to_multiple_of=pad_to_multiple_of,
        max_length=max_sequence_length
    )
    # convert features to torch tensors
    for k, v in input_features.items():
        if not isinstance(v, torch.Tensor):
            input_features[k] = torch.as_tensor(v, device=device).unsqueeze(0)

    # compute policy with offline policy model.
    logits = policy_model(input_features)

    # convert logits to probabilities
    all_probs = torch.softmax(logits, dim=-1)

    # compute top-k predictions
    topk_probs, topk_indices = torch.topk(all_probs, top_k, sorted=True)
    topk_probs = topk_probs.tolist()[0]
    topk_indices = topk_indices.tolist()[0]

    # epsilon greedy algorithm
    p = np.random.random()
    if p < epsilon:
        # a random action from top-k predictions.
        idx = np.random.choice(top_k)
    else:
        # action with the highest probability
        idx = topk_indices[0]
    # # randomly sample an action to from top_k predictions.
    # idx = random.choice(topk_indices)
    id2goal = {v: k for k, v in goal2id.items()}
    action = id2goal[idx]
    return action


def simulate_conversation(generation_model, generation_tokenizer, know_generation_model, know_tokenizer, policy_model,
                          policy_tokenizer, state, horizon=5,
                          max_sequence_length=512, max_gen_length=50, padding='max_length',
                          pad_to_multiple_of=True, goal2id=None, terminated_action=None, device=None,
                          greedy_search=True, top_k=3, epsilon=0.1, use_rtcp_policy=False, topic2id=None):
    """
    function that simulates a conversation between an user and a system starting from a given input state.
    @param generation_model: a response generation used to produce a system response
    @param generation_tokenizer: a huggingface tokenizer used for response generation
    @param know_generation_model: a knowledge generation model used to produce relevant knowledge
    @param know_tokenizer: a huggingface tokenizer used with the knowledge generation model
    @param policy_model: a prediction model used to produce a system action
    @param policy_tokenizer: a huggingface tokenizer
    @param state: the current state of the env
    @param horizon: the maximum number of turn in the simulated conversation
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param padding: type of padding
    @param pad_to_multiple_of: if pad to multiple instances.
    @param goal2id: a dictionary that convert goals to indices.
    @param device the device to allocate tensors
    @param greedy_search: True if we use greedy search
    @param top_k: get top_k predictions
    @param epsilon: a small probability used for exploration
    @return: the last generated system response.
    """
    is_terminal = False
    i = 0
    start_state = copy.deepcopy(state)
    simulated_conversation = []
    # loop for horizon rounds
    while (not is_terminal) and i < horizon:

        if greedy_search:
            # predict system action using the offline policy model
            if not use_rtcp_policy:
                action = predict_action(policy_model,
                                        policy_tokenizer,
                                        start_state,
                                        max_sequence_length,
                                        goal2id,
                                        pad_to_multiple_of,
                                        padding,
                                        device=device)
            else:
                action = predict_action_rtcp(
                    policy_model,
                    policy_tokenizer,
                    state,
                    max_sequence_length,
                    goal2id,
                    topic2id,
                    pad_to_multiple_of=True,
                    padding='max_length',
                    device=device)
        else:
            # improving the uncertainty of the predictions
            # choose one from top_k predictions
            action = predict_topk_action(policy_model,
                                         policy_tokenizer,
                                         start_state,
                                         max_sequence_length,
                                         goal2id,
                                         pad_to_multiple_of,
                                         padding,
                                         device=device,
                                         top_k=top_k,
                                         epsilon=epsilon)

        # generate relevant knowledge
        knowledge = generate_knowledge_with_plm(generation_model=know_generation_model,
                                                tokenizer=know_tokenizer,
                                                action=action,
                                                state=start_state,
                                                max_sequence_length=max_sequence_length,
                                                max_gen_length=max_gen_length,
                                                pad_to_multiple_of=pad_to_multiple_of,
                                                padding=padding,
                                                device=device)

        # generate the system response using chatgpt
        # later it will be replaced by the generated response by BART.
        # system_resp = get_user_resp(start_state, action)
        system_resp = generate_sys_response_with_plm(generation_model=generation_model,
                                                     tokenizer=generation_tokenizer,
                                                     action=action,
                                                     knowledge=knowledge,
                                                     state=start_state,
                                                     max_sequence_length=max_sequence_length,
                                                     max_gen_length=max_gen_length,
                                                     pad_to_multiple_of=pad_to_multiple_of,
                                                     padding=padding,
                                                     device=device)
        # check the terminated condition
        if check_terminated_condition(action, terminated_action):
            is_terminal = True

        # simulate user response.
        user_resp = get_user_resp(start_state, system_resp)

        # update state
        start_state = update_state(start_state, action, system_resp, user_resp)

        i += 1

        # update the simulated conversation
        simulated_conversation.extend([
            {'role': 'system', 'content': system_resp, 'goal': action},
            {'role': 'user', 'content': user_resp}
        ])

    # return the simulated conversation and the initial state.
    return simulated_conversation


# define a reward function based the generated conversation
def reward_func(conversations, target_topic, target_goal, delta=1, temperature=1):
    """
    function that computes the reward given an input generated conversation
    @param conversations: the input conversation
    @param target_topic: the target topic
    @param target_goal: the target goal
    @param delta: parameter that controls the weight of the length part
    @param temperature: temperature
    @return: a float value which is the reward.
    """
    # this is the intermediate reward during the state transition.
    reward = 0
    for utt in conversations:
        if utt['role'] == 'system':
            if target_topic.lower().strip() in utt['content'].lower().strip():
                reward = 3.0
    return reward


def compute_reward(outcome, llm_score, conv_length, alpha=3.0, lamda=1, temperature=0.5):
    """
    function that maps outcome to reward
    @param outcome: the outcome of the conversation
    @param conv_length: the length of the dialogue continuation
    @param alpha: scaling parameter of the outcome
    @param lamda: scaling parameter for the conversation length part.
    @param temperature: the temperature
    @return:
    """
    return (outcome * alpha) * llm_score + lamda * math.exp(-1.0 * conv_length / temperature)
    # no len constraint
    # return (outcome * alpha) * llm_score


# def compute_reward_based_on_memory(state, memory, pos_reward=1, neg_reward=-1, k=10):
#     """
#     function that compute the reward by using the memory
#     @param state: the input state
#     @param memory: the given memory
#     @param pos_reward: the reward value for a positive instance.
#     @param neg_reward: the reward value for a negative instance
#     @param k: number of sampled candidates
#     @return: a float which is the reward for the agent.
#     """
#     dialogue_context = state['dialogue_context']
#     dialogue_context = concatenate_sentences(dialogue_context)
#     search_args = {
#         "queries": dialogue_context,
#         "k": 1000
#     }
#     scores, indices = compute_run_time(memory.search, search_args)
#     count = 0
#     reward_scores = []
#     prob_scores = []
#     check_list = []

#     for score, idx in list(zip(scores[0], indices[0])):
#         # dialogue continuation

#         instance = memory.instances[idx]
#         if count >= k:
#             break

#         conv_id = int(instance['conv_id'])
#         if conv_id in check_list:
#             continue

#         simulated_conversation = memory.train_convs[int(instance['conv_id'])]
#         simulated_conversation = reformat_demonstration(simulated_conversation['conversation'],
#                                       is_agent_start=simulated_conversation['goal_type_list'][0] == 'Greetings')

#         llm_score = get_llm_based_assessment(target_topic=state['task_background']['target_topic'],
#                                              simulated_conversation = simulated_conversation,
#                                              demonstration=state['demonstration'], 
#                                              n = 1)
#         check = False
#         for utt in simulated_conversation:
#             if utt['role'] == "user":
#                 continue
#             if state['task_background']['target_topic'].lower() in utt['content'].lower():
#                 check = True

#         if llm_score > 0:
#             reward_scores.append(compute_reward(pos_reward, llm_score, len(instance)))
#         # failed conversations.
#         else:
#             reward_scores.append(compute_reward(neg_reward, 1 - llm_score, len(instance)))

#         # get the retrieval scores.
#         prob_scores.append(score)
#         count += 1
#         check_list.append(conv_id)

#     # compute softmax function
#     # prob_scores = softmax(np.array(prob_scores))
#     # print(prob_scores)
#     # print(reward_scores)
#     # compute the reward
#     reward = np.sum(np.array(reward_scores) * prob_scores)
#     # print(reward)
#     return reward

def compute_reward_based_on_memory(state, memory, pos_reward=1, neg_reward=-1, k=10, epsilon=1.0):
    """
    function that compute the reward by using the memory
    @param state: the input state
    @param memory: the given memory
    @param pos_reward: the reward value for a positive instance.
    @param neg_reward: the reward value for a negative instance
    @param k: number of sampled candidates
    @return: a float which is the reward for the agent.
    """
    dialogue_context = state['dialogue_context']
    dialogue_context = concatenate_sentences(dialogue_context)
    search_args = {
        "queries": dialogue_context,
        "k": k
    }
    scores, indices = compute_run_time(memory.search, search_args)
    count = 0
    reward_scores = []
    prob_scores = []
    for score, idx in list(zip(scores[0], indices[0])):
        # dialogue continuation
        continuation = memory.instances[idx]
        llm_score = memory.scores[idx]
        check = False
        for utt in continuation:
            if utt['role'] == "user":
                continue
            if state['task_background']['target_topic'].lower().strip() in utt['content'].lower():
                check = True
        # sucessful cases.
        if check and llm_score >= epsilon:
            reward_scores.append(compute_reward(pos_reward, llm_score, len(continuation)))
        # failed conversations.
        else:
            reward_scores.append(compute_reward(neg_reward, 1 - llm_score, len(continuation)))

        # get the retrieval scores.
        prob_scores.append(score)
        count += 1
        # check_list.append(conv_id)

    # compute softmax function
    prob_scores = softmax(np.array(prob_scores))
    # compute the reward
    reward = np.sum(np.array(reward_scores) * prob_scores)
    print(reward)
    return reward


def random_seed(seed):
    """
    function that init important libraries with a predefined random seed.
    @param seed: a int number
    @return: None
    """
    import random
    import torch
    import numpy as np
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def construct_initial_state(target_item, system_initial_resp="Hi !, How do I help you ?", dataset='durecdial'):
    """
    function that constructs the initial state for each conversation
    @param target_item: the targeted item
    @param system_initial_resp: default system response
    @return: the constructed state.
    """
    seed = random.randint(0, 10000)
    random_seed(seed)
    if 'conv' in target_item['demonstration']:
        del target_item['demonstration']['conv']

    if dataset == 'durecdial':
        goal = "Greetings"
        topic = "Greetings"
    else:
        goal = "no_strategy"
        topic = "no_strategy"

    state = {
        "task_background": {
            "target_topic": target_item['topic'],
            "target_goal": target_item['goal']
        },
        "demonstration": target_item["demonstration"],
        "dialogue_context": [],
        # "goal": "Greetings",  # will not affect anything, only including it for code convenience
        # "topic": "Greetings",
        "goal": goal,  # will not affect anything, only including it for code convenience
        "topic": topic,
        "knowledge": "",  # will not affect anything, only including it for code convenience
        "response": "",  # will not affect anything, only including it for code convenience
        "pre_goals": [],
        "pre_topics": []
    }
    user_initial_response = get_user_resp(state, sys_response=system_initial_resp, dataset=dataset)
    state['dialogue_context'].append({'role': 'user', 'content': user_initial_response})
    return state


def self_simulation(num_simulations, target_set, generation_model, generation_tokenizer, know_generation_model,
                    know_tokenizer, policy_model,
                    policy_tokenizer, horizon=5,
                    max_sequence_length=512, max_gen_length=50, padding='max_length',
                    pad_to_multiple_of=True, goal2id=None, terminated_action=None, device=None,
                    greedy_search=True, top_k=3, epsilon=0.1, n=5, dataset='durecdial'):
    """
    function that simulates a conversation between an user and a system starting from a given input state.
    @param num_simulations: number of simulations used to run each target item
    @param target_set the targeted item set.
    @param generation_model: a response generation used to produce a system response
    @param generation_tokenizer: a huggingface tokenizer used for response generation
    @param know_generation_model: a knowledge generation model used to produce relevant knowledge
    @param know_tokenizer: a huggingface tokenizer used with the knowledge generation model
    @param policy_model: a prediction model used to produce a system action
    @param policy_tokenizer: a huggingface tokenizer
    @param horizon: the maximum number of turn in the simulated conversation
    @param max_sequence_length: the maximum number of tokens in the input sequence.
    @param max_gen_length: the maximum number of tokens in the generated response.
    @param padding: type of padding
    @param pad_to_multiple_of: if pad to multiple instances.
    @param goal2id: a dictionary that convert goals to indices.
    @param device the device to allocate tensors
    @param greedy_search: True if we use greedy search
    @param top_k: get top_k predictions
    @param epsilon: a small probability used for exploration
    @return: a set of simulated conversations.
    """
    memory_instances = []
    for target_item in tqdm(target_set):
        for i in range(num_simulations):
            # adding some randomization
            seed = random.randint(0, 10000)
            random_seed(seed)
            # construct the initial state
            state = construct_initial_state(target_item, dataset=dataset)
            # generate a simulated conversation
            simulated_conversation = simulate_conversation(
                generation_model=generation_model,
                generation_tokenizer=generation_tokenizer,
                know_generation_model=know_generation_model,
                know_tokenizer=know_tokenizer,
                policy_model=policy_model,
                policy_tokenizer=policy_tokenizer,
                state=state,
                horizon=horizon,
                max_sequence_length=max_sequence_length,
                max_gen_length=max_gen_length,
                padding=padding,
                pad_to_multiple_of=pad_to_multiple_of,
                goal2id=goal2id,
                terminated_action=terminated_action,
                device=device,
                greedy_search=greedy_search,
                top_k=top_k,
                epsilon=epsilon
            )

            # compute LLM-based assessment
            score = get_llm_based_assessment(target_item['topic'], simulated_conversation, demonstration=None, n=n)
            # reformat the simulated conversations and store it into the memory
            memory_instances.extend(reformat_simulated_conversation([state, simulated_conversation, score]))
    return memory_instances


def reformat_simulated_conversation(simulated_conversation):
    """
    function that reformat a simulated conversation into state - continuation format
    @param simulated_conversation: a list with two elements the initial state and the simulated conversation
    @return: a list of state - continuation elements
    """
    state, conversation, score = simulated_conversation
    instances = []
    for idx, utt in enumerate(conversation):
        if utt['role'] == "user":
            state['dialogue_context'].append(utt)
            continue

        new_state = copy.deepcopy(state)
        new_state['pre_goals'].append(utt['goal'][0])
        new_state['pre_topics'].append(utt['goal'][1])
        new_state['dialogue_context'].append({'role': 'assistant', 'content': utt['content']})
        instance = {'state': new_state, 'continuation': conversation[idx:], 'score': score}
        instances.append(instance)
        state = new_state
    return instances


def get_llm_based_assessment(target_topic, simulated_conversation, demonstration=None, n=10):
    """
    function that evaluate if a target-driven conversation is successful with LLM.
    @param target_topic: the targe item
    @param simulated_conversation:
    @param demonstration: the given 1-shot demonstration.
    @param n: number of times to prompt the LLM.
    @param thresh: the threshold to determine if a conversation is successful.
    @return: 1 if the conversation is successful or 0 if it is failed.
    """
    messages = []
    if demonstration is not None:
        system_instruction_1 = ''' This is an example of a {} conversation between an user (you) and the system.
        In this conversation, the user (you) accepted  the item : {}
        '''.format(demonstration['target_goal'], demonstration['target_topic'])

        # the first instruction prompt
        messages = [
            {"role": "system", "content": system_instruction_1},
        ]
        # 1-shot demonstration
        for utt in reformat_demonstration(demonstration,
                                          is_agent_start=demonstration['goal_type_list'][0] == 'Greetings'):
            messages.append(utt)

    system_instruction_2 = """
    The following is a new conversation between a recommender and an user.
    """
    # the second instruction prompt
    messages.append(
        {"role": "system", "content": system_instruction_2},
    )
    # simulated conversation
    copied_conv = copy.deepcopy(simulated_conversation)
    for utt in copied_conv:
        # switch role
        if utt['role'] == 'system':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        temp = {'role': utt['role'], 'content': utt['content']}
        messages.append(temp)

    accept_string = "accept"
    reject_string = "reject"

    # assessment instruction
    system_instruction_3 = '''Based on the given conversation, you need to infer the attitude of the user towards the 
    target item : {}. You need to infer if the user is happy and willing to accept the target item: {}. 
    If the user is happy, you need to generate the word: {}.
    If the user is confused or not willing to accept the item :{}, you need to generate the word: {}.
    '''.format(target_topic, target_topic, accept_string, target_topic, reject_string)

    # the new generate response.
    messages.append(
        {'role': 'system', 'content': system_instruction_3}
    )

    # print(target_topic)
    # print(messages)

    responses = []
    for i in range(n):
        # getting the response.
        # response = openai.ChatCompletion.create(
        #     model=MODEL,
        #     messages=messages,
        #     temperature=1.1,
        #     max_tokens=50
        # )
        response = chat_completion_with_backoff(
            model=MODEL,
            messages=messages,
            temperature=1.1,
            max_tokens=50
        )
        responses.append(response.choices[0]['message']['content'])

    is_successful = 0
    for response in responses:
        if response.lower() == accept_string.lower():
            is_successful += 1

    return float(is_successful) / n


def get_system_response_with_LLama(state, action):
    """
    function that generate a system response with the LLama model
    @param state: the current state of the conversation
    @param action: the predicted action from the policy model which is a tuple consisting a goal and a topic.
    @return: a text string which is the system response.
    """
    demonstration = state['demonstration']['conversation']
    target_topic = state['task_background']['target_topic']
    goal, topic = action
    demonstration_context = ""
    # 1-shot demonstration
    for utt in reformat_demonstration(demonstration,
                                      is_agent_start=state['demonstration']['goal_type_list'][0] == 'Greetings'):
        demonstration_context += utt['role'] + ": " + utt['content'] + "."

    # the second instruction prompt
    # current conversation
    dialogue_context = ""
    for utt in state['dialogue_context']:
        role = utt['role']
        if utt['role'] == 'assistant':
            role = 'system'
        dialogue_context += role + ": " + utt['content'] + "."

    system_instruction_1 = '''You are playing the role of a recommender system. 
    You need to generate a response to the user.'''
    system_instruction_2 = '''You are a recommender system chatting with a user for recommendation. 
    You are playing the system role. Your target items: {}. You must follow the instructions below during chat.
    You should never directly tell the target item title.
    The following is an example conversation between a recommender (you) and an user: {}.
    You are playing the system role.
    you goal is : {}.
    you must to {} about {}
    Given the following conversation between you (system) and an user, you need to generate a response to the user: {}
    '''.format(target_topic, demonstration_context, goal, goal, topic, dialogue_context)

    output = replicate.run(
        "meta/llama-2-7b-chat:f1d50bb24186c52daae319ca8366e53debdaa9e0ae7ff976e918df752732ccc4",
        input={
            "top_p": 1,
            "prompt": system_instruction_1,
            "temperature": 0,
            "system_prompt": system_instruction_2,
            "max_new_tokens": 50,
            "repetition_penalty": 3
        }
    )
    print(output)
