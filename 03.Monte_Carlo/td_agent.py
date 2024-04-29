#
# TD Learning Agent 를 구현하세요
# 힌트 : mc_agent.py 파일을 약간 변형하면 td_agent.py 를 구현할 수 있습니다.
#


import numpy as np
import random
from collections import defaultdict
from environment import Env

# 0 이면 every-visit
# 1 이면 first-visit
EveryVisit0_FirstVisit1 = 0


# 몬테카를로 에이전트 (모든 에피소드 각의 샘플로 부터 학습)
class TDAgent:
    def __init__(self, actions):
        self.width = 5
        self.height = 5
        self.actions = actions  # 모든 상태에서 동일한 set의 행동 선택 가능
        self.learning_rate = 0.01
        self.discount_factor = 0.9
        self.epsilon = 0.1  # epsilon-Greedy 정책
        self.samples = []  # 하나의 episode 동안의 기록을 저장하기 위한 버퍼/메모리
        self.value_table = defaultdict(float)  # 가치함수를 저장하기 위한 버퍼

    # 샘플 버퍼/메모리에 샘플을 추가
    def save_sample(self, state, reward, done):
        self.samples.append([state, reward, done])

    def update(self, state, next_state, reward):
        state = str(state)
        next_state = str(next_state)
        current_value = self.value_table[state]
        next_value = self.value_table[next_state]
        new_value = current_value + self.learning_rate * (reward +
                        self.discount_factor * next_value - current_value)
        self.value_table[state] = new_value

    # 상태-가치함수에 따라서 행동을 결정
    # 다음 time-step 때 선택할 수 있는 상태들 중에서, 가장 큰 가치함수 값을 리턴하는 상태로 이동
    # 입실론 탐욕 정책을 사용
    def get_action(self, state_):
        """
            3. 여기를 구현하세요
        """
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.actions)
        else:
            next_state = self.possible_next_state(state_)
            action = self.arg_max(next_state)
        return int(action)

    # 후보가 여럿이면 arg_max를 계산하고 무작위로 하나를 반환
    # => 정책 (pi)은 없지만, 최적의 정책을 유도하는 역할을 하는 함수
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

    # 현재 상태가 state 일때, 다음 상태가 될 수 있는 모든 상태에 대한 가치함수 계산
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


# 메인 함수
if __name__ == "__main__":
    env = Env()
    agent = TDAgent(actions=list(range(env.n_actions)))

    MAX_EPISODES = 1000  # 최대 에피소드 수
    for episode in range(MAX_EPISODES):
        state = env.reset()  # 에피소드 시작 : 환경을 초기화하고, 상태 = 초기상태로 설정
        action = agent.get_action(state)

        print('Episode [' + str(episode) + '] begins at state ' + str(state))

        while True:
            env.render()
            next_state, reward, done = env.step(action)
            agent.update(state, next_state, reward)
            action = agent.get_action(state)
            state = next_state
            if done:
                print('- Episode finished')
                break

