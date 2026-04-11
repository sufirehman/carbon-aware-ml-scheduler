import numpy as np
import random

class CarbonRLAgent:
    def __init__(self, actions=[0, 1], alpha=0.1, gamma=0.9, epsilon=0.2):
        self.q_table = {}
        self.actions = actions  # 0 = run now, 1 = delay
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, carbon):
        # discretize carbon intensity
        return int(carbon // 10)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]

        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        if next_state not in self.q_table:
            self.q_table[next_state] = [0, 0]

        old_value = self.q_table[state][action]
        next_max = max(self.q_table[next_state])

        new_value = old_value + self.alpha * (reward + self.gamma * next_max - old_value)
        self.q_table[state][action] = new_value