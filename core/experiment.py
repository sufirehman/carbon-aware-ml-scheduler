from codecarbon import EmissionsTracker
import time
import numpy as np

from core.rl_agent import RLScheduler
from core.scheduler import CarbonScheduler
import pandas as pd

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

    df = pd.DataFrame({
        "from": range(len(carbon_data)),
        "to": range(1, len(carbon_data) + 1),
        "forecast": carbon_data,
        "actual": carbon_data
    })

    scheduler = CarbonScheduler(df)

    best, worst, _ = scheduler.find_optimal_window(
        duration_minutes=60,
        urgency="medium"
    )

    delay = int(max(best["delay_hours"], 0) * 3600)

    print(f"Heuristic delay: {delay} seconds")

    time.sleep(min(delay, 5))  # safety cap for testing

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
    delay = min(best_time, 5)    
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