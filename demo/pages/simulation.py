import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.carbon_api import CarbonAPI
from core.experiment import run_experiment

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
# SIDEBAR SETTINGS
# ----------------------------
st.sidebar.header("🧪 Experiment Settings")

urgency = st.sidebar.selectbox(
    "Urgency Level",
    ["low", "medium", "high"]
)

# ----------------------------
# TRAIN FUNCTION (TEMPORARY WORKLOAD)
# Replace later with your LSTM training
# ----------------------------
def train_function():
    import numpy as np

    x = np.random.rand(2000, 2000)

    for _ in range(10):
        x = x @ x


# ----------------------------
# RUN EXPERIMENT BUTTON
# ----------------------------
if st.button("🚀 Run Full Experiment"):

    # ----------------------------
    # LOAD CARBON DATA
    # ----------------------------
    api = CarbonAPI()
    df = api.get_24h_forecast()

    df["carbon"] = df["actual"].fillna(df["forecast"])
    carbon_values = df["carbon"].values

    # ----------------------------
    # RUN EXPERIMENT PIPELINE
    # ----------------------------
    results = run_experiment(carbon_values, train_function)

    # Convert results to dataframe
    results_df = pd.DataFrame([results])

    # ----------------------------
    # SHOW RESULTS
    # ----------------------------
    st.markdown("## 📊 Experiment Results")

    st.dataframe(results_df, use_container_width=True)

    # ----------------------------
    # SIMPLE VISUAL COMPARISON
    # ----------------------------
    st.markdown("## 📈 Emissions Comparison")

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Baseline",
        x=["Baseline"],
        y=[results["baseline"]]
    ))

    fig.add_trace(go.Bar(
        name="Heuristic",
        x=["Heuristic"],
        y=[results["heuristic"]]
    ))

    fig.add_trace(go.Bar(
        name="RL",
        x=["RL"],
        y=[results["rl"]]
    ))

    fig.update_layout(
        barmode="group",
        title="Carbon Emissions Comparison",
        yaxis_title="CO₂ Emissions (kg)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # KEY INSIGHT
    # ----------------------------
    best_method = min(results, key=results.get)

    st.success(
        f"""
        🏆 Best Performing Method: **{best_method.upper()}**

        This experiment demonstrates measurable differences between:
        - Baseline (no scheduling)
        - Heuristic scheduling
        - RL-based scheduling

        These results form the core empirical contribution of your paper.
        """
    )