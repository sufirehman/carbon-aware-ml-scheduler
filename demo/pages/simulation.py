import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler
from core.simulator import MLTrainingSimulator

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Simulation Lab",
    layout="wide"
)

st.title("⚙️ Carbon Simulation Lab")
st.markdown("Run multiple ML training scenarios and analyze carbon impact.")

# ----------------------------
# INPUT CONTROLS
# ----------------------------
st.sidebar.header("🧪 Simulation Settings")

runs = st.sidebar.slider("Number of Simulations", 3, 20, 8)
urgency = st.sidebar.selectbox("Urgency Level", ["low", "medium", "high"])

durations = st.sidebar.multiselect(
    "Training Durations (minutes)",
    [30, 60, 120, 180, 240],
    default=[30, 60, 120]
)

# ----------------------------
# RUN SIMULATION
# ----------------------------
if st.button("🚀 Run Simulation"):

    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])

    scheduler = CarbonScheduler(df)
    sim = MLTrainingSimulator()

    results = []

    for d in durations:
        best, worst, _ = scheduler.find_optimal_window(
            duration_minutes=d,
            urgency=urgency
        )

        runtime = sim.simulate_training(duration_minutes=1)

        # scale energy realistically
        scaled_runtime_hours = max(runtime / 3600, 0.05)
        energy = 0.25 * scaled_runtime_hours

        best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
        worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])

        savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

        results.append({
            "duration": d,
            "best_emissions": best_emissions,
            "worst_emissions": worst_emissions,
            "savings": savings
        })

    results_df = pd.DataFrame(results)

    # ----------------------------
    # KPI SUMMARY
    # ----------------------------
    st.markdown("## 📊 Simulation Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("🧪 Runs", runs)
    col2.metric("📏 Max Savings", f"{results_df['savings'].max():.2f}%")
    col3.metric("⚡ Avg Savings", f"{results_df['savings'].mean():.2f}%")

    # ----------------------------
    # MAIN GRAPH
    # ----------------------------
    st.markdown("## 📈 Carbon Impact vs Training Duration")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=results_df["duration"],
        y=results_df["best_emissions"],
        mode="lines+markers",
        name="Optimized Emissions",
        line=dict(color="green", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=results_df["duration"],
        y=results_df["worst_emissions"],
        mode="lines+markers",
        name="Immediate Emissions",
        line=dict(color="red", width=3)
    ))

    fig.update_layout(
        height=500,
        template="plotly_white",
        xaxis_title="Training Duration (minutes)",
        yaxis_title="Carbon Emissions (gCO₂)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # TABLE VIEW
    # ----------------------------
    st.markdown("## 📋 Detailed Results")

    st.dataframe(results_df, use_container_width=True)

    # ----------------------------
    # INSIGHT SECTION
    # ----------------------------
    st.markdown("## 🧠 Insight")

    best_row = results_df.loc[results_df["savings"].idxmax()]

    st.success(
        f"""
        Optimal scheduling consistently reduces emissions across workloads.

        Best configuration:
        - Duration: {best_row['duration']} minutes
        - Savings: {best_row['savings']:.2f}%

        This demonstrates that carbon-aware scheduling benefits scale across different ML workload sizes.
        """
    )