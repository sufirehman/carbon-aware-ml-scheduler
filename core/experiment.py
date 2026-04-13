from codecarbon import EmissionsTracker
import time
import numpy as np

from core.rl_agent import RLScheduler
from core.scheduler import CarbonScheduler


# -------------------------------
# 1. BASELINE (NO SCHEDULING)
# -------------------------------
def run_baseline(train_function):
    tracker = EmissionsTracker()
    
    print("\nRunning baseline (no scheduling)...")
    
    tracker.start()
    train_function()
    emissions = tracker.stop()
    
    print(f"Baseline Emissions: {emissions} kg CO2")
    
    return emissions


# -------------------------------
# 2. HEURISTIC SCHEDULER
# -------------------------------
def run_with_heuristic(carbon_data, train_function):
    print("\nRunning with heuristic scheduler...")
    
    scheduler = CarbonScheduler()
    
    # Example: scheduler decides delay
    delay = scheduler.decide(carbon_data)
    
    print(f"Heuristic delay: {delay} seconds")
    
    time.sleep(delay)
    
    tracker = EmissionsTracker()
    tracker.start()
    
    train_function()
    
    emissions = tracker.stop()
    
    print(f"Heuristic Emissions: {emissions} kg CO2")
    
    return emissions


# -------------------------------
# 3. RL SCHEDULER
# -------------------------------
def run_with_rl(carbon_data, train_function):
    print("\nRunning with RL scheduler...")
    
    # 🔥 Initialize correctly
    rl_agent = RLScheduler(carbon_data)
    
    # 🔥 Train and get best time index
    best_time = rl_agent.train()
    
    print(f"RL selected time index: {best_time}")
    
    # 🔥 Convert index → delay (important!)
    delay = best_time   # assuming each step = 1 unit time
    
    print(f"RL delay: {delay} seconds")
    
    time.sleep(delay)
    
    tracker = EmissionsTracker()
    tracker.start()
    
    train_function()
    
    emissions = tracker.stop()
    
    print(f"RL Emissions: {emissions} kg CO2")
    
    return emissions


# -------------------------------
# 4. MAIN EXPERIMENT
# -------------------------------
def run_experiment(carbon_data, train_function):
    
    baseline = run_baseline(train_function)
    heuristic = run_with_heuristic(carbon_data, train_function)
    rl = run_with_rl(carbon_data, train_function)
    
    print("\n=== FINAL RESULTS ===")
    print(f"Baseline: {baseline}")
    print(f"Heuristic: {heuristic}")
    print(f"RL: {rl}")
    
    return {
        "baseline": baseline,
        "heuristic": heuristic,
        "rl": rl
    }