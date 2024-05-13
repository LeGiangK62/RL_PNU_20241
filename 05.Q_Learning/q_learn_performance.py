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

    learning_rates = [1, 0.1, 0.001]  # Learning rates to compare
    avg_cumulative_rewards = {lr: [] for lr in learning_rates}
    exploration_rates = {lr: [] for lr in learning_rates}
    policy_qualities = {lr: [] for lr in learning_rates}

    for lr in learning_rates:
        print(f"Testing with learning rate: {lr}")
        agent = QLearningAgent(actions=actions)

        for episode in range(EPISODE_MAX):
            state = env.reset()
            cumulative_reward = 0
            epsilon = 0.1  # Fixed exploration rate for this example

            while True:
                action = agent.get_action(str(state), epsilon)
                next_state, reward, done = env.step(action)
                agent.learn(str(state), action, reward, str(next_state), lr, 0.9)

                cumulative_reward += agent.q_table[str(state)][action]
                state = next_state

                if done:
                    break

            avg_cumulative_rewards[lr].append(cumulative_reward)
            exploration_rates[lr].append(epsilon)
            policy_qualities[lr].append(np.mean(avg_cumulative_rewards[lr]))  # Average over last 100 episodes

    # Visualize the results
    import matplotlib.pyplot as plt

    episodes = range(EPISODE_MAX)
    for lr in learning_rates:
        plt.plot(episodes, avg_cumulative_rewards[lr], label=f"Learning Rate: {lr}")
    plt.xlabel("Episode")
    plt.ylabel("Average Cumulative Reward")
    plt.title("Average Cumulative Reward per Episode")
    plt.legend()
    plt.show()

    for lr in learning_rates:
        plt.plot(episodes, policy_qualities[lr], label=f"Learning Rate: {lr}")
    plt.xlabel("Episode")
    plt.ylabel("Policy Quality")
    plt.title("Policy Quality per Episode")
    plt.legend()
    plt.show()
