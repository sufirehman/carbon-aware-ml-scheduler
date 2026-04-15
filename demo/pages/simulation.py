import sys
import os
import numpy as np

# 1. Find the root of the project (Carbon-Aware-MLOps)
# This looks 2 levels up from demo/pages/
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))

# 2. Add the root to the VERY START of the python path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 3. Import using the full package path
try:
    from core.experiment import run_experiment
except ImportError:
    # Backup for different cloud structures
    sys.path.append(os.getcwd())
    from core.experiment import run_experiment

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.carbon_api import CarbonAPI
# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Carbon Simulation Lab",
    layout="wide"
)

st.title("⚙️ Carbon Simulation Lab")
st.markdown("Run baseline, heuristic, and RL experiments with real emissions measurement.")


# ----------------------------
# TRAIN FUNCTION (TEMP WORKLOAD)
# ----------------------------
def train_function():
    import numpy as np

    x = np.random.rand(4000, 4000)

    for _ in range(50):
        x = x @ x


# ----------------------------
# RUN EXPERIMENT
# ----------------------------
if st.button("🚀 Run Full Experiment"):

    progress = st.progress(0)
    status = st.empty()

    # ----------------------------
    # STEP 1: LOAD DATA
    # ----------------------------
    status.info("📡 Loading carbon forecast data...")
    progress.progress(10)

    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])

    # Add stochastic noise (for RL realism)
    base = df["carbon"].values

    trend = np.sin(np.linspace(0, 3.14, len(df))) * 4
    noise = np.random.normal(0, 6, len(df))

    df["carbon"] = base + noise + trend

    # ----------------------------
    # STEP 2: RUN EXPERIMENT
    # ----------------------------
    status.info("🧪 Running carbon-aware experiment (baseline vs heuristic vs RL)...")
    progress.progress(30)

    with st.spinner("Running experiments... please wait ⏳"):
        results = run_experiment(df, train_function, runs=5)

    progress.progress(70)

    # ----------------------------
    # STEP 3: PROCESS RESULTS
    # ----------------------------
    status.info("📊 Processing results and generating insights...")
    
    results_df = pd.DataFrame([{
        "Baseline (kg CO2)": results["baseline"],
        "Heuristic (kg CO2)": results["heuristic"],
        "RL (kg CO2)": results["rl"],

        "Baseline (g CO2)": results["baseline"] * 1000,
        "Heuristic (g CO2)": results["heuristic"] * 1000,
        "RL (g CO2)": results["rl"] * 1000,
    }])

    progress.progress(90)

    # ----------------------------
    # DONE
    # ----------------------------
    progress.progress(100)
    status.success("✅ Experiment completed successfully!")

    # ----------------------------
    # RESULTS TABLE
    # ----------------------------
    st.markdown("## 📊 Experiment Results")
    st.dataframe(results_df, use_container_width=True)

    # ----------------------------
    # CHART
    # ----------------------------
    st.markdown("## 📈 Emissions Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Baseline",
        x=["Baseline"],
        y=[results["baseline"] * 1000]
    ))

    fig.add_trace(go.Bar(
        name="Heuristic",
        x=["Heuristic"],
        y=[results["heuristic"] * 1000]
    ))

    fig.add_trace(go.Bar(
        name="RL",
        x=["RL"],
        y=[results["rl"] * 1000]
    ))

    fig.update_layout(
        barmode="group",
        title="Carbon Emissions Comparison",
        yaxis_title="CO₂ Emissions (grams)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # INSIGHT
    # ----------------------------
    clean_results = {
    "baseline": float(results["baseline"]),
    "heuristic": float(results["heuristic"]),
    "rl": float(results["rl"])
    }

    best_method = min(clean_results, key=clean_results.get)

    st.success(f"""
🏆 **Best Performing Method: {best_method.upper()}**

This demonstrates measurable carbon savings using intelligent scheduling.

✔ Baseline = no optimization  
✔ Heuristic = rule-based scheduling  
✔ RL = adaptive learning-based scheduling  
""")