from core.rl_agent import RLScheduler
import numpy as np

def simulate_rl(carbon_values, runs=50):
    emissions = []

    for _ in range(runs):
        agent = RLScheduler(carbon_values)
        best_time = agent.train()

        emissions.append(carbon_values[best_time])

    return np.mean(emissions)