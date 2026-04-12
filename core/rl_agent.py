import numpy as np
import random

class RLScheduler:
    def __init__(self, carbon_values, episodes=200):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes

        # Q-table (state = time index)
        self.q_table = np.zeros(self.n)

        # Hyperparameters
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        self.epsilon = 0.3    # exploration

    def reward(self, t):
        # Lower carbon = better reward
        return -self.carbon[t]

    def train(self):
        for _ in range(self.episodes):
            for t in range(self.n):

                # ε-greedy
                if random.random() < self.epsilon:
                    action = random.randint(0, self.n - 1)
                else:
                    action = np.argmax(self.q_table)

                r = self.reward(action)

                # Q update
                self.q_table[action] = self.q_table[action] + self.alpha * (
                    r + self.gamma * np.max(self.q_table) - self.q_table[action]
                )

        # best learned time
        best_time = np.argmax(self.q_table)
        return best_time