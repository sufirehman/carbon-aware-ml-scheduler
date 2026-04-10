import sys
import os

# FIX: correct path to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler
from core.simulator import MLTrainingSimulator
from core.report_generator import generate_report


# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Carbon-Aware ML Training Controller",
    layout="wide"
)

st.title("🌱 Carbon-Aware ML Training Controller")
st.markdown("Optimize ML training time using real UK grid carbon intensity data.")


# ----------------------------
# SESSION STATE INIT
# ----------------------------
if "results_ready" not in st.session_state:
    st.session_state.results_ready = False


# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("⚙️ Configuration")

duration = st.sidebar.slider("Training Duration (minutes)", 30, 240, 60)
urgency = st.sidebar.selectbox("Urgency Level", ["low", "medium", "high"])


# ----------------------------
# RUN PIPELINE
# ----------------------------
if st.button("🚀 Run Optimization"):

    # 1. DATA
    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])

    # 2. SCHEDULER
    scheduler = CarbonScheduler(df)
    best, worst, all_windows = scheduler.find_optimal_window(
        duration_minutes=duration,
        urgency=urgency
    )

    # 3. SIMULATION
    sim = MLTrainingSimulator()
    runtime = sim.simulate_training(duration_minutes=1)

    scaled_runtime_hours = max(runtime / 3600, 0.05)
    energy = 0.25 * scaled_runtime_hours

    best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
    worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])

    savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

    # SAVE EVERYTHING IN STATE
    st.session_state.best = best
    st.session_state.worst = worst
    st.session_state.savings = savings
    st.session_state.best_emissions = best_emissions
    st.session_state.worst_emissions = worst_emissions
    st.session_state.df = df

    st.session_state.results_ready = True


# ----------------------------
# SHOW RESULTS (ONLY IF READY)
# ----------------------------
if st.session_state.results_ready:

    best = st.session_state.best
    worst = st.session_state.worst
    savings = st.session_state.savings
    best_emissions = st.session_state.best_emissions
    worst_emissions = st.session_state.worst_emissions
    df = st.session_state.df


    # ----------------------------
    # HEADER
    # ----------------------------
    st.subheader("🌍 Carbon-Aware Insights Dashboard")


    # ----------------------------
    # KPI ROW
    # ----------------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌱 Carbon Savings", f"{savings:.2f}%")
    col2.metric("⚡ Best Emissions", f"{best_emissions:.2f} gCO₂")
    col3.metric("🔥 Worst Emissions", f"{worst_emissions:.2f} gCO₂")

    delay = abs(
        (pd.Timestamp(best["start"]) - pd.Timestamp(worst["start"]))
        .total_seconds() / 3600
    )

    col4.metric("⏱ Optimal Delay", f"{delay:.1f} hrs")


    # ----------------------------
    # INTERACTIVE GRAPH
    # ----------------------------
    st.markdown("## 📈 Grid Carbon Intensity Forecast (Interactive)")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["from"],
        y=df["carbon"],
        mode="lines",
        name="Carbon Intensity",
        line=dict(color="black", width=2)
    ))

    fig.add_vrect(
        x0=best["start"],
        x1=best["end"],
        fillcolor="green",
        opacity=0.25,
        annotation_text="Optimal Window"
    )

    fig.add_vrect(
        x0=worst["start"],
        x1=worst["end"],
        fillcolor="red",
        opacity=0.15,
        annotation_text="Worst Window"
    )

    fig.update_layout(
        height=520,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Time",
        yaxis_title="gCO₂/kWh"
    )

    st.plotly_chart(fig, use_container_width=True)


    # ----------------------------
    # COMPARISON
    # ----------------------------
    st.markdown("## 📊 Comparative Analysis")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### ❌ Run Immediately")
        st.metric("Emissions", f"{worst_emissions:.2f} gCO₂")

    with colB:
        st.markdown("### ✅ Optimized Run")
        st.metric("Emissions", f"{best_emissions:.2f} gCO₂")


    # ----------------------------
    # TABLE
    # ----------------------------
    st.markdown("## 📋 Detailed Results")

    st.dataframe(pd.DataFrame({
        "Scenario": ["Immediate", "Optimized"],
        "Carbon Intensity": [worst["avg_carbon"], best["avg_carbon"]],
        "Emissions": [worst_emissions, best_emissions],
        "Savings (%)": [0, savings]
    }))


    # ----------------------------
    # REPORT DOWNLOAD (FIXED)
    # ----------------------------
    st.markdown("## 📄 Report")

    if st.button("📄 Generate Carbon Report"):

        file = generate_report(savings, best, worst)

        with open(file, "rb") as f:
            st.download_button(
                label="⬇ Download PDF Report",
                data=f,
                file_name="carbon_report.pdf",
                mime="application/pdf"
            )


    # ----------------------------
    # INSIGHT
    # ----------------------------
    st.success(
        f"""
        🚀 Insight:
        Running ML workloads at optimal carbon windows reduces emissions by {savings:.2f}%.
        This demonstrates practical carbon-aware scheduling for AI systems.
        """
    )