import copy
import openai

API_KEY = "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"
# MODEL = "gpt-3.5-turbo"
MODEL = "text-davinci-003"
REQUEST_TIMEOUT = 20

openai.api_key = API_KEY


def convert_list_to_str(lis):
    """function that converts a list of elements to a text string

    Args:
        lis (_type_): list of lists, each element in the list is a list with 3 elements.

    Returns:
        _type_: a text string.
    """
    output_str = ""
    for k in lis:
        if len(k) > 0:
            tmp = f"{k[0]} {k[1]} {k[2]} "
            output_str += tmp
    return output_str


def convert_dialogue_history_to_str(his):
    dialogue_str = ""
    for utt in his:
        tmp = f"{utt['role']}: {utt['content']}."
        dialogue_str += tmp
    return dialogue_str

def generate_sys_resp(state, action):
    """ Generate a system response using ChatGPT.
    Args:
        state (_type_): the current state which consists of task background, pre_topics and prev_goals
        action (_type_): a chosen goal.
    """
    knowledge_str = convert_list_to_str([state['knowledge']])
    user_profile_str = convert_list_to_str(state['task_background']['user_profile'])
    target_topic = state['task_background']['target_topic']
    target_goal = state['task_background']['target_goal']
    # system_instruction = f"""You are an AI research assistant. Your action is ```{action}``. You will be given some knowledge deliminated by the following triple backticks ```{knowledge_str}```
    #     and information about the user deliminated by the following triple backticks ```{user_profile_str}```. You need to generate a response 
    #     using the action ``` {action}```, the knowledge ```{knowledge_str}``` and the user profile ```{user_profile_str}```. If the action is {target_goal}, you need to recommend the item :{target_topic}.
    # """
    dialogue_his = convert_dialogue_history_to_str(state['dialogue_context'])
    system_instruction = f"""You are a recommender chatting with an user. Given the action: {action} and the task background: {knowledge_str} and the conversation history: {dialogue_his}, 
    please generate a response. If the action: {target_goal}, you should recommend the target item {target_topic}.
    """
    # messages = [
    #     {"role": "system", "content": system_instruction},
    # ]
    # for utt in state['dialogue_context']:
    #     messages.append(utt)
    response = openai.Completion.create(
        model=MODEL,
        # messages=messages,
        prompt=system_instruction, 
        temperature=0,
        max_tokens=50,
        request_timeout = REQUEST_TIMEOUT

    )
    return response.choices[0]['text']


def get_user_resp(state, sys_response, logit_bias = None):
    if logit_bias is None:
        logit_bias = {}
    target_topic = state['task_background']['target_topic']
    seeker_instruction = '''You are a seeker chatting with a recommender for recommendation. 
    You must follow the instructions below during chat.
    You should respond to the recommender's response : {}
    If the recommender suggests the item: {}, you should accept the item. You should never directly tell the target item title.
    '''.format(target_topic, sys_response)
    response = openai.Completion.create(
        model='text-davinci-003', 
        prompt=seeker_instruction, 
        temperature=0, 
        max_tokens=50, 
        stop='Recommender',
        logit_bias=logit_bias,
        request_timeout = REQUEST_TIMEOUT

    )
    return response['choices'][0]['text']

def update_state(state, action, sys_response, user_response):
    """function that updates the state of the conversation
    in order to update the state, we need to simulate an user's response using ChatGPT.
    Args:
        state (_type_): the current state of the conversation.
        action (_type_): the chosen goal.
        response (_type_): the generated user response.
    """
    ### update state

    new_state = copy.deepcopy(state)
    new_state['dialogue_context'].append(
        {"role": "assistant", "content": sys_response}
    )
    new_state['dialogue_context'].append(
        {"role": "user", "content": user_response}
    )
    new_state['pre_goals'].append(action)
    return new_state