import numpy as np
import random


class RLScheduler:
    def __init__(self, carbon_values, episodes=1000):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes

        # Q-table (value per time index)
        self.q_table = np.zeros(self.n)

        # Hyperparameters
        self.alpha = 0.1      # learning rate
        self.gamma = 0.9      # discount factor
        self.epsilon = 0.3    # initial exploration

        # 🔥 NEW: multi-objective weights (TUNE THESE IN PAPER)
        self.lambda_delay = 0.05
        self.lambda_uncertainty = 0.5

    # ----------------------------
    # 🔥 NEW REWARD FUNCTION
    # ----------------------------
    def reward(self, t, window=3):

        # invalid window
        if t + window >= self.n:
            return -999

        segment = self.carbon[t:t + window]

        avg_carbon = np.mean(segment)
        uncertainty = np.std(segment)   # proxy for forecast uncertainty

        delay_penalty = t * self.lambda_delay

        # 🔥 multi-objective reward
        reward = -avg_carbon - (self.lambda_uncertainty * uncertainty) - delay_penalty

        return reward

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    def train(self):

        for ep in range(self.episodes):

            # epsilon decay
            self.epsilon = max(0.05, 0.3 * (1 - ep / self.episodes))

            for t in range(self.n):

                # ε-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, self.n - 1)
                else:
                    action = np.argmax(self.q_table)

                r = self.reward(action)

                # Q-learning update
                self.q_table[action] = self.q_table[action] + self.alpha * (
                    r + self.gamma * np.max(self.q_table) - self.q_table[action]
                )

        # best learned time index
        best_time = int(np.argmax(self.q_table))

        return best_time