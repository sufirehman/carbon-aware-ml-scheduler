import sys
import os
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from core.experiment import run_experiment
except ImportError:
    sys.path.append(os.getcwd())
    from core.experiment import run_experiment

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.carbon_api import CarbonAPI

st.set_page_config(
    page_title="CarbonML · Simulation Lab",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }
html, body, .stApp {
    background: #080d12 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }
.stApp::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none; z-index: 0;
}

/* ── PAGE HEADER ── */
.page-badge-amber {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #f59e0b;
    background: rgba(245,158,11,0.1);
    border: 1px solid rgba(245,158,11,0.25);
    padding: 4px 12px; border-radius: 6px;
    letter-spacing: 0.08em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-header h1 {
    font-size: 26px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.02em;
    margin: 0 0 6px;
}
.page-header p { font-size: 13px; color: #475569; margin: 0 0 24px; }

/* ── EXPERIMENT RESULT CARDS ── */
.exp-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 28px 20px;
    text-align: center;
    transition: all 0.2s;
}
.exp-card:hover {
    border-color: rgba(255,255,255,0.12);
    background: #141f2e;
}
.exp-type {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px;
}
.exp-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px; font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.exp-unit { font-size: 11px; color: #334155; margin-bottom: 14px; }
.exp-badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; padding: 4px 12px;
    border-radius: 4px; letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-worst { color: #ef4444; background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.18); }
.badge-mid   { color: #f59e0b; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.18); }
.badge-best  { color: #22c55e; background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }

/* ── CHART CARD ── */
.chart-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.chart-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #64748b;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── PROGRESS BARS ── */
.prog-section {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}
.prog-title {
    font-size: 14px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.01em;
    margin-bottom: 20px;
}
.prog-row { margin-bottom: 14px; }
.prog-label-row {
    display: flex; justify-content: space-between;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #475569;
    margin-bottom: 6px;
}
.prog-track {
    background: rgba(255,255,255,0.05);
    border-radius: 4px; height: 8px; overflow: hidden;
}
.prog-fill { height: 100%; border-radius: 4px; }

/* ── SYSTEM LOG ── */
.sys-log {
    background: #060b10;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 18px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #475569;
    line-height: 2.0; min-height: 80px;
}
.log-green { color: #22c55e; }
.log-check { color: #22c55e; font-weight: 600; }

/* ── INFO PANEL ── */
.info-panel {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 22px;
}
.info-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 16px;
}
.info-row {
    margin-bottom: 14px;
    padding-bottom: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.info-row:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.info-method {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500;
    margin-bottom: 4px;
}
.info-desc { font-size: 12px; color: #475569; line-height: 1.6; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0e1520 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }

/* ── BUTTONS ── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 8px !important;
    transition: all 0.2s !important;
    padding: 10px 24px !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"] {
    background: #22c55e !important;
    color: #071a10 !important;
    border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #16a34a !important;
    box-shadow: 0 6px 20px rgba(34,197,94,0.3) !important;
}

/* Plotly */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Progress bar override */
div[data-testid="stProgress"] > div { background: rgba(34,197,94,0.2) !important; }
div[data-testid="stProgress"] > div > div { background: #22c55e !important; }
</style>
""", unsafe_allow_html=True)

# ── PAGE HEADER
st.markdown("""
<div class="page-header">
    <div class="page-badge-amber">LAB</div>
    <h1>Carbon Simulation Lab</h1>
    <p>Run baseline, heuristic, and RL experiments with real emissions measurement</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:12px;color:#f59e0b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:16px;">⚙ Experiment Config</div>', unsafe_allow_html=True)
    runs = st.selectbox("Runs per method", [3, 5, 10], index=1,
                        help="More runs = more stable results")
    workload = st.selectbox("Workload type", ["Matrix compute", "CNN training", "Transformer fine-tune"])
    noise_level = st.slider("Carbon noise (σ)", 0, 20, 6,
                            help="Simulated grid uncertainty added to forecast")
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#334155;letter-spacing:0.06em;">EXPERIMENT INFO</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#475569;margin-top:6px;line-height:1.8">
        Baseline: no optimisation<br>
        Heuristic: rule-based delay<br>
        RL: adaptive learning agent
    </div>
    """, unsafe_allow_html=True)

# ── TRAINING FUNCTION
def train_function():
    x = np.random.rand(4000, 4000)
    for _ in range(50):
        x = x @ x

# ── LAYOUT
main_col, info_col = st.columns([3, 1])

with info_col:
    st.markdown("""
    <div class="info-panel">
        <div class="info-title">Method Guide</div>
        <div class="info-row">
            <div class="info-method" style="color:#ef4444">Baseline</div>
            <div class="info-desc">Training runs immediately with no carbon awareness. Represents current industry default.</div>
        </div>
        <div class="info-row">
            <div class="info-method" style="color:#f59e0b">Heuristic</div>
            <div class="info-desc">Rule-based scheduler delays execution to next predicted low-carbon window.</div>
        </div>
        <div class="info-row">
            <div class="info-method" style="color:#22c55e">RL Agent</div>
            <div class="info-desc">Learns optimal timing under uncertainty. Trained on historical UK grid patterns.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with main_col:

    # RUN BUTTON
    run = st.button("▶  Run Full Experiment", type="primary")

    if run:
        progress_bar = st.progress(0)
        status = st.empty()

        status.markdown('<div class="sys-log"><span class="log-green">></span> Loading 24-hour carbon forecast data...</div>', unsafe_allow_html=True)
        progress_bar.progress(10)

        api = CarbonAPI()
        df = api.get_24h_forecast()
        df["carbon"] = df["actual"].fillna(df["forecast"])
        base = df["carbon"].values
        trend = np.sin(np.linspace(0, 3.14, len(df))) * 4
        noise = np.random.normal(0, noise_level, len(df))
        df["carbon"] = base + noise + trend

        status.markdown('<div class="sys-log"><span class="log-green">></span> Running experiments (baseline vs heuristic vs RL)...</div>', unsafe_allow_html=True)
        progress_bar.progress(30)

        with st.spinner(""):
            results = run_experiment(df, train_function, runs=runs)

        progress_bar.progress(90)
        status.markdown('<div class="sys-log"><span class="log-green">></span> Processing results...</div>', unsafe_allow_html=True)
        progress_bar.progress(100)
        status.empty()
        progress_bar.empty()

        st.session_state.update({
            "sim_results": results,
            "sim_ready": True
        })

    if st.session_state.get("sim_ready"):
        results = st.session_state.sim_results

        base_g  = float(results["baseline"])  * 1000
        heur_g  = float(results["heuristic"]) * 1000
        rl_g    = float(results["rl"])        * 1000
        best    = min(base_g, heur_g, rl_g)
        worst   = max(base_g, heur_g, rl_g)

        heur_save = ((base_g - heur_g) / base_g * 100)
        rl_save   = ((base_g - rl_g)   / base_g * 100)

        # ── RESULT CARDS
        c1, c2, c3 = st.columns(3)
        card_data = [
            (c1, "#ef4444", "Baseline",  base_g, "badge-worst", "No optimisation"),
            (c2, "#f59e0b", "Heuristic", heur_g, "badge-mid",   "Rule-based"),
            (c3, "#22c55e", "RL Agent",  rl_g,   "badge-best",  "Adaptive learning"),
        ]
        for col, color, method, val, badge_cls, badge_txt in card_data:
            with col:
                st.markdown(f"""
                <div class="exp-card">
                    <div class="exp-type">{method}</div>
                    <div class="exp-val" style="color:{color}">{val:.1f}</div>
                    <div class="exp-unit">g CO₂</div>
                    <div class="exp-badge {badge_cls}">{badge_txt}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── PROGRESS COMPARISON
        st.markdown(f"""
        <div class="prog-section">
            <div class="prog-title">Relative Emissions Comparison</div>
            <div class="prog-row">
                <div class="prog-label-row">
                    <span>Baseline (no optimisation)</span>
                    <span style="color:#ef4444">100%</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:100%;background:#ef4444"></div>
                </div>
            </div>
            <div class="prog-row">
                <div class="prog-label-row">
                    <span>Heuristic (rule-based)</span>
                    <span style="color:#f59e0b">{heur_g/base_g*100:.1f}%</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:{heur_g/base_g*100}%;background:#f59e0b"></div>
                </div>
            </div>
            <div class="prog-row">
                <div class="prog-label-row">
                    <span>RL agent (adaptive)</span>
                    <span style="color:#22c55e">{rl_g/base_g*100:.1f}%</span>
                </div>
                <div class="prog-track">
                    <div class="prog-fill" style="width:{rl_g/base_g*100}%;background:#22c55e"></div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── BAR CHART
        st.markdown('<div class="chart-card"><div class="chart-title">Emissions Comparison · g CO₂</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=["Baseline", "Heuristic", "RL Agent"],
            y=[base_g, heur_g, rl_g],
            text=[f"{base_g:.1f}g", f"{heur_g:.1f}g", f"{rl_g:.1f}g"],
            textposition="auto",
            textfont=dict(family="IBM Plex Mono", size=11),
            marker_color=["#ef4444", "#f59e0b", "#22c55e"],
            marker_line_width=0,
            width=0.45
        ))
        fig.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, color="#475569",
                       tickfont=dict(family="IBM Plex Mono", size=12)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                       color="#334155", tickfont=dict(family="IBM Plex Mono", size=10),
                       title="g CO₂"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Determine best method
        methods = {"Baseline": base_g, "Heuristic": heur_g, "RL Agent": rl_g}
        best_method = min(methods, key=methods.get)

        # ── LOG
        st.markdown(f"""
        <div class="sys-log">
            <span class="log-green">></span> Carbon forecast loaded · noise σ={noise_level} added for realism<br>
            <span class="log-green">></span> Baseline: {base_g:.2f}g CO₂ (no optimisation)<br>
            <span class="log-green">></span> Heuristic: {heur_g:.2f}g CO₂ ({heur_save:.1f}% saving vs baseline)<br>
            <span class="log-green">></span> RL agent: {rl_g:.2f}g CO₂ ({rl_save:.1f}% saving vs baseline)<br>
            <span class="log-check">✓</span> Best performing method: <strong style="color:#22c55e">{best_method.upper()}</strong> — demonstrating measurable carbon savings via intelligent scheduling
        </div>
        """, unsafe_allow_html=True)

    else:
        # Empty state
        st.markdown("""
        <div style="
            background:#0e1520;
            border:1px solid rgba(255,255,255,0.07);
            border-radius:14px;
            padding:60px 40px;
            text-align:center;
            margin-top:8px;
        ">
            <div style="font-size:32px;margin-bottom:16px;opacity:0.3">🤖</div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#334155;letter-spacing:0.04em;">
                Configure your experiment in the sidebar and run to compare methods
            </div>
        </div>
        """, unsafe_allow_html=True)