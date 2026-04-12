import numpy as np
import random

class RLScheduler:
    def __init__(self, carbon_values, episodes=1000):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes

        # Q-table (state = time index)
        self.q_table = np.zeros(self.n)

        # Hyperparameters
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        # self.epsilon = 0.3    # exploration

    def reward(self, t, window=3):
        if t + window >= len(self.carbon):
            return -999  # invalid

        avg = sum(self.carbon[t:t+window]) / window

        return -avg

    def train(self):
        for ep in range(self.episodes):

            # 🔥 EPSILON DECAY (THIS IS WHERE IT GOES)
            self.epsilon = max(0.05, 0.3 * (1 - ep / self.episodes))

            for t in range(self.n):

                # ε-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, self.n - 1)
                else:
                    action = np.argmax(self.q_table)

                # reward from chosen action
                r = self.reward(action)

                # Q-learning update
                self.q_table[action] = self.q_table[action] + self.alpha * (
                    r + self.gamma * np.max(self.q_table) - self.q_table[action]
                )

        # best learned time index
        best_time = np.argmax(self.q_table)
        return best_time