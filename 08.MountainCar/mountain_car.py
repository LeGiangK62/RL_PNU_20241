import gym
import pylab
import random
import numpy as np
from collections import deque
from keras.layers import Dense
from keras.optimizers import Adam
from keras.models import Sequential

EPISODES = 1000


class DQNAgent:
    def __init__(self, state_size, action_size):

        self.render = True

        self.state_size = state_size
        self.action_size = action_size

        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_min = 0.005
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / 50000
        self.batch_size = 64
        self.train_start = 1000
        self.memory = deque(maxlen=10000)

        self.model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()

    def build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(64, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(32, activation='relu', kernel_initializer='he_uniform'))
        model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
        model.summary()
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            q_value = self.model.predict(state)
            return np.argmax(q_value[0])

    def replay_memory(self, state, action, reward, next_state, done):
        if action == 2:
            action = 1
        self.memory.append((state, action, reward, next_state, done))
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

    def train_replay(self):
        if len(self.memory) < self.train_start:
            return
        batch_size = min(self.batch_size, len(self.memory))
        mini_batch = random.sample(self.memory, batch_size)

        update_input = np.zeros((batch_size, self.state_size))
        update_target = np.zeros((batch_size, self.action_size))

        for i in range(batch_size):
            state, action, reward, next_state, done = mini_batch[i]
            target = self.model.predict(state)[0]

            if done:
                target[action] = reward
            else:
                target[action] = reward + self.discount_factor * \
                                          np.amax(self.target_model.predict(next_state)[0])
            update_input[i] = state
            update_target[i] = target

        self.model.fit(update_input, update_target, batch_size=batch_size, epochs=1, verbose=0)

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('MountainCar-v0')
    state_size = env.observation_space.shape[0]
    action_size = 2

    agent = DQNAgent(state_size, action_size)
    # agent.load_model("save_model/MountainCar_DQN.h5")
    scores, episodes = [], []

    for e in range(EPISODES):
        done = False
        score = 0
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        print(state)

        fake_action = 0

        action_count = 0

        while not done:
            if agent.render:
                env.render()

            # 현재 상태에서 행동을 선택하고 한 스텝을 진행
            action_count = action_count + 1

            if action_count == 4:
                action = agent.get_action(state)
                action_count = 0

                if action == 0:
                    fake_action = 0
                elif action == 1:
                    fake_action = 2

            # 선택한 액션으로 1 step을 시행한다
            next_state, reward, done, info = env.step(fake_action)
            next_state = np.reshape(next_state, [1, state_size])
            # 에피소드를 끝나게 한 행동에 대해서 -100의 패널티를 줌
            #reward = reward if not done else -100

            # <s, a, r, s'>을 replay memory에 저장
            agent.replay_memory(state, fake_action, reward, next_state, done)
            # 매 타임스텝마다 학습을 진행
            agent.train_replay()
            score += reward
            state = next_state

            if done:
                env.reset()
                agent.update_target_model()

                scores.append(score)
                episodes.append(e)
                pylab.plot(episodes, scores, 'b')
                pylab.savefig("save_graph/MountainCar_DQN.png")
                print("episode:", e, "  score:", score, "  memory length:", len(agent.memory),
                      "  epsilon:", agent.epsilon)

        # 50 에피소드마다 학습 모델을 저장
        if e % 50 == 0:
            agent.save_model("save_model/MountainCar_DQN.h5")

