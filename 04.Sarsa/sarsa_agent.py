import numpy as np
import random
from collections import defaultdict
from env import Env

MAX_EPISODE = 100

class SARSAgent:
    def __init__(self, actions):
        self.actions = actions
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 1  # 3) 시간이 지날수록 e 값이 감소하도록 코드를 수정하세요.
        self.eps_step = 0.9
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    # 큐함수 업데이트
    def learn(self, cur_s, cur_a, r, next_s, next_a):
        """
            2) 여기에 들어갈 내용을 구현하세요
        """
        current_q = self.q_table[cur_s][cur_a]
        next_state_q = self.q_table[next_s][next_a]
        new_q = (current_q + self.learning_rate *
                 (r + self.discount_factor * next_state_q - current_q))
        self.q_table[cur_s][cur_a] = new_q

    # 입실론 탐욕 정책에 따라서 행동을 반환
    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            # 무작위 행동 선택 (exploration)
            best_action = np.random.choice(self.actions)
        else:
            # 큐함수에 따른 최적 행동 반환 (exploitation)
            state_action = self.q_table[state]
            best_action = self.arg_max(state_action)
        return best_action

    """
    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = state_action[0]
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)
    """

    @staticmethod
    def arg_max(state_action):
        max_index_list = []
        max_value = -9999
        for index, value in enumerate(state_action):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)

    def decaying_epsilon(self):
        self.epsilon = self.epsilon * self.eps_step


if __name__ == "__main__":
    env = Env()  # 환경에 대한 instance 생성
    agent = SARSAgent(actions=list(range(env.n_actions)))  # Sarsa Agent 객체 생성

    # 지정된 횟수(MAX_EPISODE)만큼 episode 진행
    for episode in range(MAX_EPISODE):
        # 게임 환경과 상태를 초기화 하고, 상태(state)값 얻기
        state = env.reset()

        # 현재 상태에서 어떤 행동을 할지 선택
        action = agent.get_action(str(state))

        print('Episode [' + str(episode) + '] begins at state ' + str(state))

        if episode % 5 == 0:
            print(f'Runing with eps = {agent.epsilon}')

        # 한개의 episode를 처음부터 끝까지 처리하는 while-loop
        while True:
            env.render()

            """
                1) 여기에 들어갈 내용을 구현하세요.
            """
            next_state, reward, done = env.step(action)

            if agent.epsilon > 0.001:
                agent.decaying_epsilon()

            next_action = agent.get_action(str(next_state))
            agent.learn(str(state), action, reward, str(next_state), next_action)

            state = next_state
            action = next_action

            # 모든 큐함수 값을 화면에 표시
            env.print_value_all(agent.q_table)

            # episode가 끝났으면 while-loop을 종료
            if done:
                break

