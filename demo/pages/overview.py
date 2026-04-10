import sys
import os
# Add the parent directory (Carbon-Aware-MLOps) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("⚙️ Configuration")

duration = st.sidebar.slider("Training Duration (minutes)", 30, 240, 60)
urgency = st.sidebar.selectbox("Urgency Level", ["low", "medium", "high"])


# ----------------------------
# RUN PIPELINE
# ----------------------------
if st.button("🚀 Run Optimization"):

    # =========================
    # 1. DATA LAYER
    # =========================
    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])

    scheduler = CarbonScheduler(df)
    best, worst, all_windows = scheduler.find_optimal_window(
        duration_minutes=duration,
        urgency=urgency
    )

    sim = MLTrainingSimulator()
    runtime = sim.simulate_training(duration_minutes=1)

    # 🔥 FIX: scale runtime to realistic ML training energy
    scaled_runtime_hours = max(runtime / 3600, 0.05)  # minimum 3 minutes equivalent
    energy = 0.25 * scaled_runtime_hours  # assume GPU load baseline

    best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
    worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])

    savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

    if st.button("📄 Download Carbon Report"):
        file = generate_report(savings, best, worst)

        with open(file, "rb") as f:
            st.download_button(
                label="Download PDF",
                data=f,
                file_name="carbon_report.pdf",
                mime="application/pdf"
            )
    # =========================
    # 2. DASHBOARD HEADER (NEW)
    # =========================
    st.subheader("🌍 Carbon-Aware Insights Dashboard")

    # =========================
    # 3. TOP KPI ROW (LIKE REAL DASHBOARDS)
    # =========================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌱 Carbon Savings", f"{savings:.2f}%")
    col2.metric("⚡ Best Emissions", f"{best_emissions:.2f} gCO₂")
    col3.metric("🔥 Worst Emissions", f"{worst_emissions:.2f} gCO₂")
    col4.metric("⏱ Optimal Delay", f"{abs((pd.Timestamp(best['start']) - pd.Timestamp(worst['start'])).total_seconds()/3600):.1f} hrs")


    # =========================
    # 4. MAIN CHART (CENTERPIECE)
    # =========================
    st.markdown("## 📈 Grid Carbon Intensity Forecast")

    # =========================
    # 4. MAIN CHART (CENTERPIECE) - PLOTLY VERSION
    # =========================

    st.markdown("## 📈 Grid Carbon Intensity Forecast (Interactive)")

    fig = go.Figure()

    # Main carbon line
    fig.add_trace(go.Scatter(
        x=df["from"],
        y=df["carbon"],
        mode="lines",
        name="Carbon Intensity (gCO₂/kWh)",
        line=dict(color="black", width=2)
    ))

    # Best window (green highlight)
    fig.add_vrect(
        x0=best["start"],
        x1=best["end"],
        fillcolor="green",
        opacity=0.25,
        line_width=0,
        annotation_text="Optimal Window",
        annotation_position="top left"
    )

    # Worst window (red highlight)
    fig.add_vrect(
        x0=worst["start"],
        x1=worst["end"],
        fillcolor="red",
        opacity=0.15,
        line_width=0,
        annotation_text="Worst Window",
        annotation_position="top left"
    )

    # Layout styling
    fig.update_layout(
        height=520,
        template="plotly_white",
        margin=dict(l=40, r=40, t=40, b=40),
        xaxis_title="Time",
        yaxis_title="Carbon Intensity (gCO₂/kWh)",
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(fig, use_container_width=True)


    # =========================
    # 5. ANALYSIS SECTION
    # =========================
    st.markdown("## 📊 Comparative Analysis")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### ❌ Run Immediately")
        st.metric("Emissions", f"{worst_emissions:.2f} gCO₂")

    with colB:
        st.markdown("### ✅ Optimized Run")
        st.metric("Emissions", f"{best_emissions:.2f} gCO₂")


    # =========================
    # 6. TABLE VIEW (ENGINEERING STYLE)
    # =========================
    st.markdown("## 📋 Detailed Results")

    st.dataframe(
        pd.DataFrame({
            "Scenario": ["Immediate", "Optimized"],
            "Carbon Intensity": [worst["avg_carbon"], best["avg_carbon"]],
            "Emissions": [worst_emissions, best_emissions],
            "Savings (%)": [0, savings]
        }),
        use_container_width=True
    )


    # =========================
    # 7. INSIGHT PANEL (IMPORTANT)
    # =========================
    st.success(
        f"""
        🚀 Insight:
        Running ML workloads at optimal carbon windows reduces emissions by {savings:.2f}%.
        This demonstrates workload scheduling as a practical decarbonization strategy for ML systems.
        """
    )