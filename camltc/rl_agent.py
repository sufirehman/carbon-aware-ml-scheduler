import numpy as np
import random


class RLScheduler:
    def __init__(self, carbon_values, episodes=8000, max_delay=20):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes
        self.max_delay = max_delay

        # Q-table: state x action (0 = wait, 1 = execute)
        self.q_table = np.zeros((self.n, 2))

        # hyperparameters
        self.alpha = 0.1
        self.gamma = 0.95
        self.epsilon = 0.4

        # reward weights (tuned for carbon-aware scheduling)
        self.w_carbon = 1.0
        self.w_uncertainty = 0.6
        self.w_delay = 0.05
        self.reward_scale = 10.0  # stability improvement

    # ----------------------------
    # CONTEXT WINDOW (LOCAL GRID SIGNAL)
    # ----------------------------
    def get_context(self, t):
        window = self.carbon[max(0, t - 3):min(self.n, t + 3)]
        return np.mean(window), np.std(window)

    # ----------------------------
    # ENVIRONMENT STEP FUNCTION
    # ----------------------------
    def step(self, state, action, delay_count):

        mean_c, std_c = self.get_context(state)

        global_mean = np.mean(self.carbon)
        peak_penalty = max(0, mean_c - global_mean)

        # ----------------------------
        # ACTION 0: WAIT
        # ----------------------------
        if action == 0:
            next_state = min(state + 1, self.n - 1)
            delay_count += 1

            reward = -self.w_delay * delay_count
            reward = reward / self.reward_scale

            done = False
            return next_state, reward, delay_count, done

        # ----------------------------
        # ACTION 1: EXECUTE
        # ----------------------------
        else:
            reward = (
                -self.w_carbon * mean_c
                -self.w_uncertainty * std_c
                -self.w_delay * delay_count
                -2.0 * peak_penalty
            )

            reward = reward / self.reward_scale

            done = True
            return state, reward, delay_count, done

    # ----------------------------
    # TRAINING LOOP (Q-LEARNING)
    # ----------------------------
    def train(self):

        for ep in range(self.episodes):

            state = random.randint(0, self.n // 2)
            delay_count = 0
            done = False

            # epsilon decay
            self.epsilon = max(0.05, 0.4 * (1 - ep / self.episodes))

            while not done:

                # ε-greedy action selection
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, delay_count, done = self.step(
                    state, action, delay_count
                )

                # Q-learning update
                best_next = np.max(self.q_table[next_state])

                self.q_table[state, action] += self.alpha * (
                    reward + self.gamma * best_next - self.q_table[state, action]
                )

                state = next_state

                if state >= self.n - 1:
                    break

        # ----------------------------
        # POLICY EXECUTION (INFERENCE)
        # ----------------------------
        state = 0
        delay_count = 0

        while state < self.n:

            action = np.argmax(self.q_table[state])

            if action == 1:
                return state  # execute time chosen

            state += 1
            delay_count += 1

        return self.n - 1