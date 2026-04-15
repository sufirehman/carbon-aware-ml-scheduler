import numpy as np
import random

# sequential decision-making 
# trade-off: wait vs run 
# uncertainty handling

class RLScheduler:
    def __init__(self, carbon_values, episodes=5000):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes

        # Q-table: state (time) x action (wait/run)
        self.q_table = np.zeros((self.n, 2))

        # Hyperparameters
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.3

        # Multi-objective weights (IMPORTANT for paper)
        self.lambda_delay = 0.02
        self.lambda_uncertainty = 0.3

    # ----------------------------
    # STEP FUNCTION (ENVIRONMENT)
    # ----------------------------
    def step(self, state, action):

        # action = 0 → WAIT
        if action == 0:
            next_state = min(state + 1, self.n - 1)

            reward = -self.lambda_delay  # penalty for waiting
            done = False

        # action = 1 → EXECUTE
        else:
            if state >= self.n:
                return state, -999, True

            carbon = self.carbon[state]

            # local uncertainty (window-based)
            window = self.carbon[max(0, state - 2): min(self.n, state + 2)]
            uncertainty = np.std(window)

            reward = -carbon - (self.lambda_uncertainty * uncertainty)
            next_state = state
            done = True

        return next_state, reward, done

    # ----------------------------
    # TRAINING LOOP
    # ----------------------------
    def train(self):

        for ep in range(self.episodes):

            state = 0
            done = False

            # epsilon decay
            self.epsilon = max(0.05, 0.3 * (1 - ep / self.episodes))

            while not done:

                # ε-greedy
                if random.random() < self.epsilon:
                    action = random.randint(0, 1)
                else:
                    action = np.argmax(self.q_table[state])

                next_state, reward, done = self.step(state, action)

                # Q update
                self.q_table[state, action] += self.alpha * (
                    reward
                    + self.gamma * np.max(self.q_table[next_state])
                    - self.q_table[state, action]
                )

                state = next_state

                # safety break
                if state >= self.n - 1:
                    break

        # ----------------------------
        # INFERENCE (policy execution)
        # ----------------------------
        state = 0

        while state < self.n:

            action = np.argmax(self.q_table[state])

            if action == 1:
                return state  # EXECUTE HERE

            state += 1

        return self.n - 1  # fallback