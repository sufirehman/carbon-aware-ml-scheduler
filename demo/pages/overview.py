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

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CarbonML · Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL STYLE (CLEANED)
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    background: #080d12 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #e2e8f0;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding: 2rem 2.2rem !important;
    max-width: 100% !important;
}

/* GRID BACKGROUND */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.02) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* =========================
   HEADER
========================= */
.page-header {
    margin-bottom: 24px;
}

.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

.page-header h1 {
    font-size: 28px;
    margin: 0;
}

.page-header p {
    font-size: 13px;
    color: #64748b;
}

/* =========================
   CARDS (UNIFIED SYSTEM)
========================= */
.card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
    transition: 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    border-color: rgba(34,197,94,0.25);
}

/* KPI */
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px;
    font-weight: 600;
    color: #f1f5f9;
}

.kpi-label {
    font-size: 11px;
    color: #64748b;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* =========================
   HERO RESULT
========================= */
.hero-box {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(14,165,233,0.05));
    border: 1px solid rgba(34,197,94,0.2);
    border-radius: 14px;
    padding: 26px;
    text-align: center;
    margin-bottom: 20px;
}

.hero-box h2 {
    font-family: 'IBM Plex Mono', monospace;
    color: #22c55e;
    margin: 0;
    font-size: 34px;
}

/* =========================
   SYSTEM LOG
========================= */
.log {
    background: #060b10;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 16px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: #64748b;
    line-height: 1.8;
}

/* BUTTONS */
div.stButton > button {
    border-radius: 8px !important;
    font-family: 'IBM Plex Mono' !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="page-header">
    <div class="page-badge">DASHBOARD</div>
    <h1>Carbon Intelligence Dashboard</h1>
    <p>Real-time ML workload optimisation using UK grid carbon intensity</p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙ Settings")

    duration = st.slider("Training Duration (min)", 30, 240, 60, 15)
    urgency = st.selectbox("Urgency", ["low", "medium", "high"], index=1)

    st.markdown("---")
    st.markdown("**Data Source**")
    st.caption("National Grid ESO API")

# =========================
# RUN BUTTON
# =========================
run = st.button("▶ Run Optimisation", type="primary")

if run:

    status = st.empty()

    status.markdown('<div class="log">Loading carbon forecast...</div>', unsafe_allow_html=True)

    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])

    scheduler = CarbonScheduler(df)

    best, worst, _ = scheduler.find_optimal_window(
        duration_minutes=duration,
        urgency=urgency
    )

    sim = MLTrainingSimulator()
    runtime = sim.simulate_training(duration_minutes=1)
    energy = 0.25 * max(runtime / 3600, 0.05)

    best_em = sim.calculate_emissions(energy, best["avg_carbon"])
    worst_em = sim.calculate_emissions(energy, worst["avg_carbon"])

    savings = ((worst_em - best_em) / worst_em) * 100

    status.empty()

    # =========================
    # HERO RESULT
    # =========================
    st.markdown(f"""
    <div class="hero-box">
        <h2>↓ {savings:.1f}% Emissions Reduction</h2>
        <p>Shift execution by optimal window for lower carbon intensity</p>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # KPIS
    # =========================
    c1, c2, c3, c4 = st.columns(4)

    kpis = [
        ("Savings", f"{savings:.1f}%"),
        ("Optimised", f"{best_em:.2f}g"),
        ("Baseline", f"{worst_em:.2f}g"),
        ("Delay", f"{abs((pd.Timestamp(best['start']) - pd.Timestamp(worst['start'])).total_seconds()/3600):.1f}h")
    ]

    for col, (label, value) in zip([c1, c2, c3, c4], kpis):
        with col:
            st.markdown(f"""
            <div class="card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-val">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    # =========================
    # CHARTS
    # =========================
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=df["from"],
            y=df["carbon"],
            mode="lines",
            line=dict(width=2),
            fill="tozeroy"
        ))

        fig.add_vrect(
            x0=best["start"],
            x1=best["end"],
            fillcolor="rgba(34,197,94,0.12)",
            line_width=0
        )

        fig.update_layout(
            template="plotly_dark",
            height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="log">', unsafe_allow_html=True)
        st.markdown(f"""
        <b>Optimal:</b> {best['start']} → {best['avg_carbon']:.1f} gCO₂/kWh<br>
        <b>Baseline:</b> {worst['start']} → {worst['avg_carbon']:.1f} gCO₂/kWh<br><br>
        <span style="color:#22c55e">✓ {savings:.2f}% emissions reduction achieved</span>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Click **Run Optimisation** to generate results.")