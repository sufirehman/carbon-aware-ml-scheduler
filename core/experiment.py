from codecarbon import EmissionsTracker
import time
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
def run_with_heuristic(df, train_function):
    from core.scheduler import CarbonScheduler   # 🔥 moved here

    print("\nRunning with heuristic scheduler...")

    scheduler = CarbonScheduler(df)
    best, worst, _ = scheduler.find_optimal_window(
        duration_minutes=60,
        urgency="medium"
    )

    delay = int(max(best["delay_hours"], 0) * 3600)

    time.sleep(min(delay, 5))

    tracker = EmissionsTracker()
    tracker.start()
    train_function()
    return tracker.stop()


# -------------------------------
# 3. RL SCHEDULER
# -------------------------------
def run_with_rl(df, train_function):
    from core.rl_agent import RLScheduler   # 🔥 moved here

    print("\nRunning with RL scheduler...")

    carbon_values = df["carbon"].values
    rl_agent = RLScheduler(carbon_values)

    best_time = rl_agent.train()

    delay = min(int(best_time), 5)

    time.sleep(delay)

    tracker = EmissionsTracker()
    tracker.start()
    train_function()
    return tracker.stop()


# -------------------------------
# 4. MAIN EXPERIMENT
# -------------------------------
def run_experiment(df, train_function, runs=3):

    baseline_list = []
    heuristic_list = []
    rl_list = []

    for _ in range(runs):

        baseline = run_baseline(train_function)
        heuristic = run_with_heuristic(df, train_function)
        rl = run_with_rl(df, train_function)

        baseline_list.append(baseline)
        heuristic_list.append(heuristic)
        rl_list.append(rl)

    return {
        "baseline": sum(baseline_list) / runs,
        "heuristic": sum(heuristic_list) / runs,
        "rl": sum(rl_list) / runs
    }