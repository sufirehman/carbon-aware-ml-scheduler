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
# 3. RL SCHEDULER (IMPROVED)
# -------------------------------
def run_with_rl(df, train_function):
    from core.rl_agent import RLScheduler

    print("\nRunning with RL scheduler (risk-aware)...")

    carbon_values = df["carbon"].values

    rl_agent = RLScheduler(carbon_values, episodes=8000)
    best_time = rl_agent.train()

    # 🔥 convert index → realistic delay
    delay_seconds = int(best_time * 60)   # assuming 1 index = 1 minute
    delay_seconds = min(delay_seconds, 30)  # keep demo fast

    time.sleep(delay_seconds)

    tracker = EmissionsTracker()
    tracker.start()
    train_function()
    emissions = tracker.stop()

    print(f"RL Emissions: {emissions} kg CO2")

    return emissions


# -------------------------------
# 4. MAIN EXPERIMENT
# -------------------------------
def run_experiment(df, train_function, runs=5):

    records = []

    for i in range(runs):

        print(f"\n--- RUN {i+1} ---")

        baseline = run_baseline(train_function)
        heuristic = run_with_heuristic(df, train_function)
        rl = run_with_rl(df, train_function)

        records.append({
            "run": i + 1,
            "baseline": baseline,
            "heuristic": heuristic,
            "rl": rl
        })

    # convert to dataframe
    results_df = pd.DataFrame(records)

    # save to CSV (🔥 THIS IS THE KEY PART)
    results_df.to_csv("emissions.csv", index=False)

    # return both mean + raw data
    return {
    "baseline": float(results_df["baseline"].mean()),
    "heuristic": float(results_df["heuristic"].mean()),
    "rl": float(results_df["rl"].mean()),
    "raw": results_df
    }