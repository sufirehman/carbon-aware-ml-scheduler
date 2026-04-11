import numpy as np
from core.rl_agent import CarbonRLAgent

def simulate_rl(carbon_data):
    agent = CarbonRLAgent()
    total_emissions = 0

    for i in range(len(carbon_data) - 1):
        state = agent.get_state(carbon_data[i])
        action = agent.choose_action(state)

        if action == 0:  # run now
            emission = carbon_data[i]
        else:  # delay
            emission = carbon_data[i + 1]

        reward = -emission
        next_state = agent.get_state(carbon_data[i + 1])

        agent.update(state, action, reward, next_state)
        total_emissions += emission

    return total_emissions