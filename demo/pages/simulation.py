import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.experiment import run_experiment
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.carbon_api import CarbonAPI

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="CarbonML · Simulation Lab",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL STYLE (SIMPLIFIED)
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
}

/* HEADER */
.badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #f59e0b;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    padding: 4px 12px;
    border-radius: 6px;
    display: inline-block;
    letter-spacing: 0.08em;
    margin-bottom: 10px;
}

/* CARDS */
.card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
}

/* EXP CARDS */
.exp {
    text-align: center;
    padding: 22px;
    background: #0e1520;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.07);
}

.exp-val {
    font-family: 'IBM Plex Mono';
    font-size: 32px;
    font-weight: 600;
}

.exp-label {
    font-size: 11px;
    color: #64748b;
    text-transform: uppercase;
}

/* LOG */
.log {
    background: #060b10;
    border: 1px solid rgba(255,255,255,0.06);
    padding: 16px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono';
    font-size: 12px;
    color: #64748b;
    line-height: 1.8;
}

/* BUTTON */
div.stButton > button {
    font-family: 'IBM Plex Mono' !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div>
    <div class="badge">SIMULATION LAB</div>
    <h1 style="margin:0">Carbon Experiment Engine</h1>
    <p style="color:#64748b;font-size:13px">
        Compare Baseline vs Heuristic vs RL scheduling strategies
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙ Config")

    runs = st.selectbox("Runs per method", [3, 5, 10], index=1)
    workload = st.selectbox(
        "Workload",
        ["Matrix compute", "CNN training", "Transformer fine-tune"]
    )
    noise = st.slider("Carbon noise (σ)", 0, 20, 6)

    st.markdown("---")
    st.caption("Baseline | Heuristic | Reinforcement Learning")

# =========================
# TRAIN FUNCTION
# =========================
def train_function():
    x = np.random.rand(4000, 4000)
    for _ in range(50):
        x = x @ x

# =========================
# RUN EXPERIMENT
# =========================
run = st.button("▶ Run Experiment", type="primary")

if run:

    progress = st.progress(0)
    log = st.empty()

    log.markdown('<div class="log">Loading carbon forecast...</div>', unsafe_allow_html=True)
    progress.progress(20)

    api = CarbonAPI()
    df = api.get_24h_forecast()

    df["carbon"] = df["actual"].fillna(df["forecast"])
    df["carbon"] += np.random.normal(0, noise, len(df))

    progress.progress(50)
    log.markdown('<div class="log">Running experiments...</div>', unsafe_allow_html=True)

    results = run_experiment(df, train_function, runs=runs)

    progress.progress(100)
    log.empty()
    progress.empty()

    st.session_state["results"] = results
    st.session_state["ready"] = True

# =========================
# RESULTS
# =========================
if st.session_state.get("ready"):

    r = st.session_state["results"]

    base = float(r["baseline"]) * 1000
    heur = float(r["heuristic"]) * 1000
    rl = float(r["rl"]) * 1000

    best = min(base, heur, rl)
    best_name = ["Baseline", "Heuristic", "RL"][np.argmin([base, heur, rl])]

    # HERO SUMMARY
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg, rgba(34,197,94,0.1), rgba(14,165,233,0.05));
        border:1px solid rgba(34,197,94,0.2);
        padding:24px;
        border-radius:12px;
        text-align:center;
        margin-bottom:20px;
    ">
        <h2 style="color:#22c55e;margin:0">{best_name} Wins</h2>
        <p style="color:#64748b;margin:0">Lowest carbon emissions achieved across experiments</p>
    </div>
    """, unsafe_allow_html=True)

    # CARDS
    c1, c2, c3 = st.columns(3)

    data = [
        ("Baseline", base, "#ef4444"),
        ("Heuristic", heur, "#f59e0b"),
        ("RL Agent", rl, "#22c55e"),
    ]

    for col, (name, val, color) in zip([c1, c2, c3], data):
        with col:
            st.markdown(f"""
            <div class="exp">
                <div class="exp-label">{name}</div>
                <div class="exp-val" style="color:{color}">{val:.1f}</div>
                <div class="exp-label">g CO₂</div>
            </div>
            """, unsafe_allow_html=True)

    # CHART
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=["Baseline", "Heuristic", "RL"],
        y=[base, heur, rl],
        marker_color=["#ef4444", "#f59e0b", "#22c55e"]
    ))

    fig.update_layout(
        template="plotly_dark",
        height=300,
        paper_bgcolor="rgba(0,0,0,0)"
    )

    st.plotly_chart(fig, use_container_width=True)

    # LOGS
    st.markdown(f"""
    <div class="log">
        Baseline: {base:.2f} g CO₂<br>
        Heuristic: {heur:.2f} g CO₂<br>
        RL Agent: {rl:.2f} g CO₂<br><br>
        <b style="color:#22c55e">Best Method: {best_name}</b>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Run simulation to start experiments")