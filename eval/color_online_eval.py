import os
import copy
from eval.base import BaseOnlineEval
from baselines.color.utils import predict_action_color

from dyna_gym.envs.utils import generate_knowledge_with_plm, \
    generate_sys_response_with_plm


class ColorBartOnlineEval(BaseOnlineEval):
    """
    Online evaluation class for RTCP with Bart knowledge and text generation model
    """

    def __init__(self, target_set, terminal_act, use_llm_score, n, k, epsilon, use_demonstration, generation_model,
                 generation_tokenizer,
                 know_generation_model,
                 know_generation_tokenizer,
                 policy_model, policy_tokenizer, horizon, goal2id, topic2id, device=None,
                 max_sequence_length=512, pad_to_multiple_of=True, padding='max_length',
                 max_gen_length=50, dataset='durecdial', args=None, is_test=True
                 ):
        """
        constructor for class MCTSCRSOnlineEval
        @param target_set:
        @param generation_model:
        @param generation_tokenizer:
        @param know_generation_model:
        @param know_generation_tokenizer:
        @param policy_model:
        @param policy_tokenizer:
        @param horizon:
        @param goal2id:
        @param device:
        @param max_sequence_length:
        @param max_gen_length:
        """

        super().__init__(target_set, terminal_act, horizon, use_llm_score, epsilon, n, use_demonstration, k,
                         dataset=dataset)
        self.generation_model = generation_model
        self.generation_tokenizer = generation_tokenizer
        self.know_generation_model = know_generation_model
        self.know_generation_tokenizer = know_generation_tokenizer
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.topic2id = topic2id
        self.horizon = horizon
        self.goal2id = goal2id
        self.device = device
        self.max_sequence_length = max_sequence_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.max_gen_length = max_gen_length
        self.args = args
        self.is_test = is_test

    def init_state(self, target_item, system_initial_resp="Hello ! "):
        state = super().init_state(target_item, system_initial_resp)
        # construct reversed goals and topics
        # just for convenience
        state['reversed_goals'] = []
        state['reversed_topics'] = []
        state['pre_goals'] = ["None"]
        state['pre_topics'] = ["None"]
        state['goal_path'] = ["None"]
        state['topic_path'] = ["None"]
        state['knowledge'] = [target_item['topic'], "", ""]
        return state

    def update(self, state, system_response, system_action, user_response):
        # update state
        new_state = copy.deepcopy(state)
        new_state['dialogue_context'].append(
            {"role": "assistant", "content": system_response}
        )
        new_state['dialogue_context'].append(
            {"role": "user", "content": user_response}
        )
        goal, topic = system_action
        new_state['pre_goals'].append(goal)
        new_state['pre_topics'].append(topic)
        new_state['goal_path'].append(goal)
        new_state['topic_path'].append(topic)
        return new_state

    def check_terminated_condition(self, system_action):
        """
        function that check if the conversation is terminated
        @param system_action: the input system action (goal)
        @return: True if the conversation is terminated else False
        """
        return system_action[0] == self.terminal_act

    def pipeline(self, state):
        """
        method that perform one system pipeline including action prediction, knowledge generation and response generation
        @param state: the current state of the conversation
        @return: generated system response and predicted system action
        """
        # predict action with color policy
        action = predict_action_color(args=self.args,
                                      policy_model=self.policy_model,
                                      policy_tokenizer=self.policy_tokenizer,
                                      state=state,
                                      max_sequence_length=self.max_sequence_length,
                                      device=self.device,
                                      is_test=self.is_test
                                      )

        # generate relevant knowledge
        knowledge = generate_knowledge_with_plm(generation_model=self.know_generation_model,
                                                tokenizer=self.know_generation_tokenizer,
                                                action=action,
                                                state=state,
                                                max_sequence_length=self.max_sequence_length,
                                                max_gen_length=self.max_gen_length,
                                                pad_to_multiple_of=self.pad_to_multiple_of,
                                                padding=self.padding,
                                                device=self.device)

        # assign the knowledge to target topic
        # just for convenience
        # state['knowledge'] = ["", "", ""]

        # generate the system response using chatgpt
        # later it will be replaced by the generated response by BART.
        system_resp = generate_sys_response_with_plm(generation_model=self.generation_model,
                                                     tokenizer=self.generation_tokenizer,
                                                     action=action,
                                                     knowledge=knowledge,
                                                     state=state,
                                                     max_sequence_length=self.max_sequence_length,
                                                     max_gen_length=self.max_gen_length,
                                                     pad_to_multiple_of=self.pad_to_multiple_of,
                                                     padding=self.padding,
                                                     device=self.device,
                                                     dataset=self.dataset
                                                     )

        return system_resp, action
