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

st.set_page_config(
    page_title="CarbonML · Dashboard",
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
.page-header { margin-bottom: 28px; }
.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #22c55e;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    padding: 4px 12px; border-radius: 6px;
    letter-spacing: 0.08em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-header h1 {
    font-size: 26px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.02em;
    margin: 0 0 6px;
}
.page-header p { font-size: 13px; color: #475569; margin: 0; }

/* ── RESULT HERO BANNER ── */
.result-hero {
    background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(14,165,233,0.06));
    border: 1px solid rgba(34,197,94,0.15);
    border-radius: 14px;
    padding: 30px 36px;
    text-align: center;
    margin-bottom: 24px;
}
.result-hero h2 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px; font-weight: 600;
    color: #22c55e; margin: 0 0 8px;
    letter-spacing: -0.02em;
}
.result-hero p { font-size: 14px; color: #64748b; margin: 0; }
.result-hero strong { color: #94a3b8; }

/* ── KPI CARDS ── */
.kpi-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px;
    padding: 20px;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.kpi-green::after  { background: #22c55e; }
.kpi-blue::after   { background: #0ea5e9; }
.kpi-red::after    { background: #ef4444; }
.kpi-amber::after  { background: #f59e0b; }

.kpi-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 8px;
}
.kpi-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 28px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.kpi-sub { font-size: 11px; color: #334155; }

/* ── CHART CARDS ── */
.chart-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 24px;
}
.chart-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #64748b;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.chart-dot {
    width: 7px; height: 7px; border-radius: 50%;
    display: inline-block;
}

/* ── SYSTEM LOG ── */
.sys-log {
    background: #060b10;
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 18px 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #475569;
    line-height: 1.9;
}
.log-green { color: #22c55e; }
.log-check { color: #22c55e; font-weight: 600; }
.log-warn  { color: #f59e0b; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #0e1520 !important;
    border-right: 1px solid rgba(255,255,255,0.07) !important;
}
section[data-testid="stSidebar"] * { color: #94a3b8 !important; }
.sidebar-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #22c55e !important;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 16px;
}

/* ── CONTROLS ── */
div[data-testid="stSlider"] > div { color: #94a3b8 !important; }
div[data-testid="stSelectbox"] label { color: #64748b !important; font-size: 12px !important; }

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
div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.2) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* Plotly chart background fix */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── PAGE HEADER
st.markdown("""
<div class="page-header">
    <div class="page-badge">DASHBOARD</div>
    <h1>Carbon Intelligence Dashboard</h1>
    <p>Real-time optimisation of ML workloads using UK grid carbon data</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR CONTROLS
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙ Configuration</div>', unsafe_allow_html=True)
    duration = st.slider("Training Duration (minutes)", 30, 240, 60, step=15)
    urgency = st.selectbox("Urgency Level", ["low", "medium", "high"], index=1)
    region = st.selectbox("UK Region", ["National", "South England", "North England", "Scotland"])
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#334155;letter-spacing:0.06em;">DATA SOURCE</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#475569;margin-top:4px;">National Grid ESO<br>Carbon Intensity API</div>', unsafe_allow_html=True)

# ── RUN BUTTON
col_btn, col_gap = st.columns([1, 3])
with col_btn:
    run = st.button("▶  Run Optimisation", type="primary")

if run:
    with st.spinner(""):
        status = st.empty()
        status.markdown('<div class="sys-log"><span class="log-green">></span> Loading 24-hour carbon forecast from National Grid ESO...</div>', unsafe_allow_html=True)

        api = CarbonAPI()
        df = api.get_24h_forecast()
        df["carbon"] = df["actual"].fillna(df["forecast"])

        scheduler = CarbonScheduler(df)
        best, worst, _ = scheduler.find_optimal_window(
            duration_minutes=duration,
            urgency=urgency
        )

        status.markdown('<div class="sys-log"><span class="log-green">></span> Running carbon simulation...</div>', unsafe_allow_html=True)

        sim = MLTrainingSimulator()
        runtime = sim.simulate_training(duration_minutes=1)
        runtime_h = max(runtime / 3600, 0.05)
        energy = 0.25 * runtime_h

        best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
        worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])
        savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

        status.empty()

        st.session_state.update({
            "df": df, "best": best, "worst": worst,
            "savings": savings,
            "best_emissions": best_emissions,
            "worst_emissions": worst_emissions,
            "ready": True
        })

if st.session_state.get("ready"):
    df            = st.session_state.df
    best          = st.session_state.best
    worst         = st.session_state.worst
    savings       = st.session_state.savings
    best_em       = st.session_state.best_emissions
    worst_em      = st.session_state.worst_emissions
    delay_h       = abs((pd.Timestamp(best["start"]) - pd.Timestamp(worst["start"])).total_seconds() / 3600)

    # ── RESULT HERO
    st.markdown(f"""
    <div class="result-hero">
        <h2>↓ {savings:.1f}% Emissions Reduction</h2>
        <p>Delay execution by <strong>{delay_h:.1f} hours</strong> to shift into a lower-carbon grid window</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        (k1, "kpi-green",  "Carbon Savings",      f"{savings:.2f}%",   "vs immediate execution"),
        (k2, "kpi-blue",   "Optimised Emissions",  f"{best_em:.2f} g",  "at best window"),
        (k3, "kpi-red",    "Baseline Emissions",   f"{worst_em:.2f} g", "if run immediately"),
        (k4, "kpi-amber",  "Recommended Delay",    f"{delay_h:.1f} hr", "to optimal window"),
    ]
    for col, cls, label, val, sub in kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card {cls}">
                <div class="kpi-label">{label}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ROW
    ch1, ch2 = st.columns([2, 1])

    with ch1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title"><span class="chart-dot" style="background:#22c55e"></span>24-HR CARBON DECISION TIMELINE</div>', unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["from"], y=df["carbon"],
            mode="lines", name="Carbon Intensity",
            line=dict(color="#64748b", width=2),
            fill="tozeroy",
            fillcolor="rgba(100,116,139,0.05)"
        ))
        fig.add_vrect(
            x0=best["start"], x1=best["end"],
            fillcolor="rgba(34,197,94,0.12)", line_width=0,
            annotation_text="Optimal", annotation_font_color="#22c55e",
            annotation_font_size=11
        )
        fig.add_vrect(
            x0=worst["start"], x1=worst["end"],
            fillcolor="rgba(239,68,68,0.08)", line_width=0,
            annotation_text="Now", annotation_font_color="#ef4444",
            annotation_font_size=11
        )
        fig.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, color="#334155", tickfont=dict(family="IBM Plex Mono", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
                       tickfont=dict(family="IBM Plex Mono", size=10), title="gCO₂/kWh"),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title"><span class="chart-dot" style="background:#0ea5e9"></span>EMISSIONS COMPARISON</div>', unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Immediate", "Optimised"],
            y=[worst_em, best_em],
            text=[f"{worst_em:.2f}g", f"{best_em:.2f}g"],
            textposition="auto",
            textfont=dict(family="IBM Plex Mono", size=11),
            marker_color=["#ef4444", "#22c55e"],
            width=0.5
        ))
        fig2.update_layout(
            height=280, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, color="#334155", tickfont=dict(family="IBM Plex Mono", size=11)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
                       tickfont=dict(family="IBM Plex Mono", size=10), title="gCO₂"),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── SYSTEM LOG
    st.markdown(f"""
    <div class="sys-log">
        <span class="log-green">></span> Carbon forecast loaded · {len(df)} intervals · 30-min resolution<br>
        <span class="log-green">></span> Optimal window: {best['start']} → avg {best['avg_carbon']:.1f} gCO₂/kWh<br>
        <span class="log-green">></span> Baseline window: {worst['start']} → avg {worst['avg_carbon']:.1f} gCO₂/kWh<br>
        <span class="log-check">✓</span> Optimisation complete · <strong style="color:#22c55e">{savings:.2f}% emissions reduction</strong> achieved by delaying {delay_h:.1f} hours
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── REPORT
    rpt_col, _ = st.columns([1, 3])
    with rpt_col:
        if st.button("Generate PDF Report", type="secondary"):
            file = generate_report(savings, best, worst)
            with open(file, "rb") as f:
                st.download_button(
                    "⬇  Download Report",
                    f, file_name="carbonml_report.pdf"
                )

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
        <div style="font-size:32px;margin-bottom:16px;opacity:0.4">⚡</div>
        <div style="font-family:'IBM Plex Mono',monospace;font-size:13px;color:#334155;letter-spacing:0.04em;">
            Configure settings in the sidebar and run optimisation to see results
        </div>
    </div>
    """, unsafe_allow_html=True)