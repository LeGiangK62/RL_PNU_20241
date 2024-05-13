import numpy as np
import random
from env import Env
from collections import defaultdict


class QLearningAgent:
    def __init__(self, actions):
        self.actions = actions
        self.q_table = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])

    def learn(self, state, action, reward, next_state, learning_rate, discount_factor):
        cur_q = self.q_table[state][action]
        new_q = reward + discount_factor * max(self.q_table[next_state])
        self.q_table[state][action] += learning_rate * (new_q - cur_q)

    def get_action(self, state, epsilon):
        if np.random.rand() < epsilon:
            action = np.random.choice(self.actions)
        else:
            state_action = self.q_table[state]
            action = self.arg_max(state_action)
        return action

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


if __name__ == "__main__":
    env = Env()
    actions = list(range(env.n_actions))
    EPISODE_MAX = 100

    discounts = [1, 0.9, 0.5, 0.1]  # Discount factors to compare
    avg_cumulative_rewards = {dis: [] for dis in discounts}
    exploration_rates = {dis: [] for dis in discounts}
    policy_qualities = {dis: [] for dis in discounts}

    for dis in discounts:
        print(f"Testing with discount factor: {dis}")
        agent = QLearningAgent(actions=actions)

        for episode in range(EPISODE_MAX):
            state = env.reset()
            cumulative_reward = 0
            epsilon = 0.1  # Fixed exploration rate for this example

            while True:
                action = agent.get_action(str(state), epsilon)
                next_state, reward, done = env.step(action)
                agent.learn(str(state), action, reward, str(next_state), 0.05, dis)

                cumulative_reward += agent.q_table[str(state)][action]
                state = next_state

                if done:
                    break

            avg_cumulative_rewards[dis].append(cumulative_reward)
            exploration_rates[dis].append(epsilon)
            policy_qualities[dis].append(np.mean(avg_cumulative_rewards[dis]))  # Average over last 100 episodes

    # Visualize the results
    import matplotlib.pyplot as plt

    episodes = range(EPISODE_MAX)
    for dis in discounts:
        plt.plot(episodes, avg_cumulative_rewards[dis], label=f"Discount Factor: {dis}")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Average Cumulative Reward per Episode")
    plt.legend()
    plt.show()

    for dis in discounts:
        plt.plot(episodes, policy_qualities[dis], label=f"Learning Rate: {dis}")
    plt.xlabel("Episode")
    plt.ylabel("Policy Quality")
    plt.title("Policy Quality per Episode")
    plt.legend()
    plt.show()
