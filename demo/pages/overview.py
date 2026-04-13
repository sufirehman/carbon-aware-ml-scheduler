import sys
import os

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
    page_title="Carbon Intelligence Dashboard",
    layout="wide"
)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:48px; background:linear-gradient(90deg,#00C9FF,#92FE9D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;">
🌍 Carbon Intelligence Dashboard
</h1>
<p style="color:#94a3b8; font-size:18px;">
Real-time optimization of ML workloads using UK grid carbon intelligence
</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# SIDEBAR
# ----------------------------
st.sidebar.header("⚙️ Configuration")

duration = st.sidebar.slider("Training Duration (minutes)", 30, 240, 60)
urgency = st.sidebar.selectbox("Urgency Level", ["low", "medium", "high"])

# ----------------------------
# RUN BUTTON
# ----------------------------
if st.button("🚀 Run Optimization"):

    with st.spinner("Analyzing carbon patterns and optimizing workload..."):

        # DATA
        api = CarbonAPI()
        df = api.get_24h_forecast()
        df["carbon"] = df["actual"].fillna(df["forecast"])

        # SCHEDULER
        scheduler = CarbonScheduler(df)
        best, worst, _ = scheduler.find_optimal_window(
            duration_minutes=duration,
            urgency=urgency
        )

        # SIMULATION
        sim = MLTrainingSimulator()
        runtime = sim.simulate_training(duration_minutes=1)

        runtime_h = max(runtime / 3600, 0.05)
        energy = 0.25 * runtime_h

        best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
        worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])

        savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

        st.session_state.update({
            "df": df,
            "best": best,
            "worst": worst,
            "savings": savings,
            "best_emissions": best_emissions,
            "worst_emissions": worst_emissions,
            "ready": True
        })

# ----------------------------
# RESULTS
# ----------------------------
if st.session_state.get("ready"):

    df = st.session_state.df
    best = st.session_state.best
    worst = st.session_state.worst
    savings = st.session_state.savings
    best_emissions = st.session_state.best_emissions
    worst_emissions = st.session_state.worst_emissions

    delay = abs(
        (pd.Timestamp(best["start"]) - pd.Timestamp(worst["start"]))
        .total_seconds() / 3600
    )

    # ----------------------------
    # 🔥 HERO INSIGHT
    # ----------------------------
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, rgba(0,201,255,0.15), rgba(146,254,157,0.1));
        padding: 28px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        text-align:center;
        margin-top:20px;
    ">
        <h2 style="margin-bottom:10px;">🚀 {savings:.1f}% Emissions Reduction Achieved</h2>
        <p style="color:#94a3b8; font-size:16px;">
        Delay execution by <b>{delay:.1f} hours</b> to shift into a lower-carbon grid window.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ----------------------------
    # KPI CARDS
    # ----------------------------
    st.markdown("## 📊 Optimization Metrics")

    def card(title, value, color):
        st.markdown(f"""
        <div style="
            background: linear-gradient(180deg, #0b1220, #070b14);
            padding:18px;
            border-radius:14px;
            border:1px solid rgba(255,255,255,0.06);
        ">
            <div style="color:#94a3b8; font-size:12px; letter-spacing:0.08em;">
                {title}
            </div>
            <div style="font-size:30px; font-weight:800; color:{color}; margin-top:6px;">
                {value}
            </div>
        </div>
        """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        card("Carbon Savings", f"{savings:.2f}%", "#22c55e")

    with c2:
        card("Optimized Emissions", f"{best_emissions:.2f} g", "#38bdf8")

    with c3:
        card("Immediate Emissions", f"{worst_emissions:.2f} g", "#ef4444")

    with c4:
        card("Delay Applied", f"{delay:.1f} hrs", "#f59e0b")

    # ----------------------------
    # 📈 DECISION GRAPH (DIFFERENT FROM FORECAST)
    # ----------------------------
    st.markdown("## 📈 Decision Timeline")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["from"],
        y=df["carbon"],
        mode="lines",
        line=dict(color="#94a3b8", width=2),
        name="Carbon Trend"
    ))

    fig.add_vrect(
        x0=best["start"],
        x1=best["end"],
        fillcolor="#22c55e",
        opacity=0.25,
        annotation_text="Optimal"
    )

    fig.add_vrect(
        x0=worst["start"],
        x1=worst["end"],
        fillcolor="#ef4444",
        opacity=0.15,
        annotation_text="Immediate"
    )

    fig.update_layout(
        height=450,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Time",
        yaxis_title="gCO₂/kWh"
    )

    st.plotly_chart(fig, use_container_width=True)

    # ----------------------------
    # ⚖️ COMPARISON CHART
    # ----------------------------
    st.markdown("## ⚖️ Strategy Comparison")

    fig2 = go.Figure()

    fig2.add_trace(go.Bar(
        x=["Immediate", "Optimized"],
        y=[worst_emissions, best_emissions],
        text=[f"{worst_emissions:.2f}", f"{best_emissions:.2f}"],
        textposition="auto"
    ))

    fig2.update_layout(
        template="plotly_dark",
        height=400,
        yaxis_title="gCO₂"
    )

    st.plotly_chart(fig2, use_container_width=True)

    # ----------------------------
    # REPORT
    # ----------------------------
    st.markdown("## 📄 Report")

    if st.button("Generate PDF Report"):
        file = generate_report(savings, best, worst)

        with open(file, "rb") as f:
            st.download_button(
                "Download Report",
                f,
                file_name="carbon_report.pdf"
            )

    # ----------------------------
    # FINAL INSIGHT
    # ----------------------------
    st.success(
        f"Delaying execution reduces emissions by {savings:.2f}% — demonstrating real-world carbon-aware AI scheduling."
    )