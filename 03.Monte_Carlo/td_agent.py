#
# TD Learning Agent 를 구현하세요
# 힌트 : mc_agent.py 파일을 약간 변형하면 td_agent.py 를 구현할 수 있습니다.
#
import numpy as np
import random
from collections import defaultdict
from environment import Env


class TDAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions  # 모든 상태에서 동일한 set의 행동 선택 가능
        self.learning_rate = 0.001
        self.discount_factor = 0.9
        self.epsilon = 0.4  # epsilon-Greedy 정책
        self.value_table = defaultdict(float)  # 가치함수를 저장하기 위한 버퍼

        # Initialize value function with random numbers
        for col in range(self.width):
            for row in range(self.height):
                self.value_table[str([col, row])] = 0

    def update(self, cur_s, next_s, r):
        current_value = self.value_table[str(cur_s)]
        next_value = self.value_table[str(next_s)]

        new_value = current_value + self.learning_rate * (r +
                        self.discount_factor * next_value - current_value)

        self.value_table[str(cur_s)] = new_value

    def update_last(self, cur_s, r):
        current_value = self.value_table[str(cur_s)]

        new_value = current_value + self.learning_rate * r

        self.value_table[str(cur_s)] = new_value


    def get_action(self, state_):
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            next_state = self.possible_next_state(state_)
            action = self.arg_max(next_state)
        return int(action)

    def possible_next_state(self, state):
        col, row = state
        next_state = [0.0] * 4

        if row != 0:
            next_state[0] = self.value_table[str([col, row - 1])]
        else:
            next_state[0] = self.value_table[str(state)]

        if row != self.height - 1:
            next_state[1] = self.value_table[str([col, row + 1])]
        else:
            next_state[1] = self.value_table[str(state)]

        if col != 0:
            next_state[2] = self.value_table[str([col - 1, row])]
        else:
            next_state[2] = self.value_table[str(state)]

        if col != self.width - 1:
            next_state[3] = self.value_table[str([col + 1, row])]
        else:
            next_state[3] = self.value_table[str(state)]

        return next_state

    @staticmethod
    def arg_max(next_state):
        max_index_list = []
        max_value = next_state[0]
        for index, value in enumerate(next_state):
            if value > max_value:
                max_index_list.clear()
                max_value = value
                max_index_list.append(index)
            elif value == max_value:
                max_index_list.append(index)
        return random.choice(max_index_list)


# 메인 함수
if __name__ == "__main__":
    env = Env()
    agent = TDAgent(actions=list(range(env.n_actions)))

    MAX_EPISODES = 1000  # 최대 에피소드 수
    for episode in range(MAX_EPISODES):

        state = env.reset()
        action = agent.get_action(state)

        print('Episode [' + str(episode) + '] begins at state ' + str(state))

        while True:
            env.render()
            next_state, reward, done = env.step(action)
            next_action = agent.get_action(next_state)
            agent.update(state, next_state, reward)
            state = next_state
            action = next_action

            if done:
                agent.update_last(state, reward)
                print('- Episode finished')
                env.print_values(agent.value_table)
                break

