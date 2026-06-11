import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from camltc.carbon_api import CarbonAPI
from camltc.scheduler import CarbonScheduler
from core.simulator import MLTrainingSimulator
from core.report_generator import generate_report

st.set_page_config(
    page_title="CAML-TC · Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# STYLES
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #05090f !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #cbd5e1;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }
.stApp::before {
    content: ''; position: fixed; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 52px 52px;
}

/* ── PAGE HEADER ── */
.page-hdr { margin-bottom: 28px; }
.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 4px 14px; border-radius: 6px;
    letter-spacing: 0.1em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-hdr h1 {
    font-size: 26px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.02em; margin: 0 0 6px;
}
.page-hdr p { font-size: 13px; color: #475569; margin: 0; }

/* ── RESULT HERO ── */
.result-hero {
    background: linear-gradient(135deg, rgba(34,197,94,0.07) 0%, rgba(14,165,233,0.04) 100%);
    border: 1px solid rgba(34,197,94,0.14);
    border-radius: 14px; padding: 36px 44px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 24px;
}
.hero-main { }
.hero-saving {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 52px; font-weight: 600;
    color: #22c55e; letter-spacing: -0.03em;
    line-height: 1; margin-bottom: 8px;
    text-shadow: 0 0 32px rgba(34,197,94,0.2);
}
.hero-label { font-size: 14px; color: #64748b; }
.hero-label strong { color: #94a3b8; }
.hero-meta {
    text-align: right;
    font-family: 'IBM Plex Mono', monospace;
}
.hero-window {
    font-size: 13px; color: #22c55e; margin-bottom: 6px;
}
.hero-window-label {
    font-size: 10px; color: #334155;
    text-transform: uppercase; letter-spacing: 0.1em;
}
.strategy-badge {
    display: inline-block; margin-top: 10px;
    font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; padding: 4px 12px;
    border-radius: 4px;
}
.strategy-rl  { color: #22c55e; background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
.strategy-heu { color: #f59e0b; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); }

/* ── KPI CARDS ── */
.kpi { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 10px; padding: 20px; position: relative; overflow: hidden; }
.kpi::after { content:''; position:absolute; bottom:0; left:0; right:0; height:2px; }
.kpi-g::after { background: #22c55e; }
.kpi-b::after { background: #0ea5e9; }
.kpi-r::after { background: #ef4444; }
.kpi-a::after { background: #f59e0b; }
.kpi-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; color:#f1f5f9; letter-spacing:-0.02em; margin-bottom:4px; }
.kpi-sub { font-size:11px; color:#475569; }

/* ── CHART CARD ── */
.chart-card { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 24px; }
.chart-hdr { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.chart-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; display:flex; align-items:center; gap:8px; }
.cdot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.chart-meta { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#1e293b; }

/* ── SYSTEM LOG ── */
.sys-log {
    background: #040810;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 20px 22px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #334155; line-height: 2.1;
}
.log-g { color: #22c55e; }
.log-ok { color: #22c55e; font-weight: 600; }
.log-ts { color: #1e293b; margin-right: 8px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #08111c !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #64748b !important; }
.sb-title { font-family:'IBM Plex Mono',monospace; font-size:11px; color:#22c55e !important; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:16px; }

/* ── BUTTONS ── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; border-radius: 8px !important;
    padding: 10px 24px !important; width: 100% !important;
    letter-spacing: 0.04em !important; transition: all 0.2s !important;
}
div.stButton > button[kind="primary"] {
    background: #22c55e !important; color: #051a0d !important; border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #16a34a !important;
    box-shadow: 0 6px 22px rgba(34,197,94,0.28) !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #475569 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important; border-color: rgba(255,255,255,0.2) !important;
}
.js-plotly-plot .plotly .bg { fill: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── PAGE HEADER
st.markdown("""
<div class="page-hdr">
    <div class="page-badge">Dashboard</div>
    <h1>Carbon Intelligence Dashboard</h1>
    <p>Real-time carbon-aware scheduling for ML workloads · UK National Grid ESO · 30-min resolution</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown('<div class="sb-title">⚙ Configuration</div>', unsafe_allow_html=True)
    duration = st.slider("Training Duration (min)", 30, 240, 60, step=15)
    urgency = st.selectbox("Urgency Level", ["low", "medium", "high"], index=1,
        help="Low = maximise savings · High = run as soon as clean window appears")
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#1e293b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">Data Source</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#334155;line-height:1.8;">National Grid ESO<br>Carbon Intensity API<br>api.carbonintensity.org.uk</div>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#1e293b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">Urgency Guide</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#334155;line-height:2.0;">low → max delay, max savings<br>medium → balanced default<br>high → first clean window</div>', unsafe_allow_html=True)

# ── RUN BUTTON
col_btn, _ = st.columns([1, 4])
with col_btn:
    run = st.button("▶  Run Optimisation", type="primary")

if run:
    with st.spinner("Querying National Grid ESO..."):
        api = CarbonAPI()
        df = api.get_24h_forecast()
        df["carbon"] = df["actual"].fillna(df["forecast"])

        scheduler = CarbonScheduler(df)
        best, worst, all_windows = scheduler.find_optimal_window(
            duration_minutes=duration, urgency=urgency
        )

        sim = MLTrainingSimulator()
        runtime = sim.simulate_training(duration_minutes=1)
        runtime_h = max(runtime / 3600, 0.05)
        energy = 0.25 * runtime_h

        best_emissions  = sim.calculate_emissions(energy, best["avg_carbon"])
        worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])
        savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

        # Determine strategy used
        volatility = float(np.std(df["carbon"].dropna().values))
        strategy = "rl" if volatility > 30 else "heuristic"

        st.session_state.update({
            "df": df, "best": best, "worst": worst,
            "savings": savings, "best_em": best_emissions,
            "worst_em": worst_emissions, "strategy": strategy,
            "all_windows": all_windows, "ready": True
        })

if st.session_state.get("ready"):
    df        = st.session_state.df
    best      = st.session_state.best
    worst     = st.session_state.worst
    savings   = st.session_state.savings
    best_em   = st.session_state.best_em
    worst_em  = st.session_state.worst_em
    strategy  = st.session_state.strategy
    delay_h   = abs((pd.Timestamp(best["start"]) - pd.Timestamp(worst["start"])).total_seconds() / 3600)

    # ── RESULT HERO
    s_cls  = "strategy-rl" if strategy == "rl" else "strategy-heu"
    s_txt  = "RL Agent" if strategy == "rl" else "Heuristic"
    st.markdown(f"""
    <div class="result-hero">
        <div class="hero-main">
            <div class="hero-saving">↓ {savings:.1f}%</div>
            <div class="hero-label">
                emissions reduction vs immediate execution ·
                delay by <strong>{delay_h:.1f} hours</strong> to shift into a greener grid window
            </div>
        </div>
        <div class="hero-meta">
            <div class="hero-window-label">Optimal Start</div>
            <div class="hero-window">{best['start'].strftime('%H:%M UTC') if hasattr(best['start'], 'strftime') else best['start']}</div>
            <div class="hero-window-label" style="margin-top:10px;">Strategy Used</div>
            <div class="strategy-badge {s_cls}">{s_txt}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI ROW
    k1, k2, k3, k4 = st.columns(4)
    for col, cls, lbl, val, sub in [
        (k1, "kpi-g", "CO₂ Saved",         f"{savings:.1f}%",    "vs immediate execution"),
        (k2, "kpi-b", "Optimal Emissions",  f"{best_em:.3f} g",   "at best window"),
        (k3, "kpi-r", "Baseline Emissions", f"{worst_em:.3f} g",  "if run now"),
        (k4, "kpi-a", "Delay Required",     f"{delay_h:.1f} hr",  "to optimal window"),
    ]:
        with col:
            st.markdown(f"""<div class="kpi {cls}">
                <div class="kpi-lbl">{lbl}</div>
                <div class="kpi-val">{val}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS
    ch1, ch2 = st.columns([2, 1])

    with ch1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("""<div class="chart-hdr">
            <div class="chart-title"><span class="cdot" style="background:#22c55e"></span>24-HR CARBON DECISION TIMELINE</div>
            <div class="chart-meta">gCO₂/kWh · 30-min intervals · hover for values</div>
        </div>""", unsafe_allow_html=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["from"], y=df["carbon"],
            mode="lines", name="Carbon Intensity",
            line=dict(color="#475569", width=1.5),
            fill="tozeroy", fillcolor="rgba(71,85,105,0.06)",
            hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f} gCO₂/kWh<extra></extra>"
        ))
        # shade all windows by score (light)
        for _, row in st.session_state.all_windows.iterrows():
            norm = 1 - min(row["score"] / st.session_state.all_windows["score"].max(), 1)
            if norm > 0.7:
                fig.add_vrect(
                    x0=row["start"], x1=row["end"],
                    fillcolor=f"rgba(34,197,94,{norm*0.06:.2f})", line_width=0
                )

        fig.add_vrect(x0=best["start"], x1=best["end"],
            fillcolor="rgba(34,197,94,0.12)", line_color="rgba(34,197,94,0.3)", line_width=1,
            annotation_text="✓ Optimal", annotation_font_color="#22c55e", annotation_font_size=11
        )
        fig.add_vrect(x0=worst["start"], x1=worst["end"],
            fillcolor="rgba(239,68,68,0.07)", line_color="rgba(239,68,68,0.2)", line_width=1,
            annotation_text="✗ Now", annotation_font_color="#ef4444", annotation_font_size=11
        )
        fig.update_layout(
            height=290, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, color="#334155", tickfont=dict(family="IBM Plex Mono", size=10)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
                       tickfont=dict(family="IBM Plex Mono", size=10), title="gCO₂/kWh"),
            hovermode="x unified", showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with ch2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown("""<div class="chart-hdr">
            <div class="chart-title"><span class="cdot" style="background:#0ea5e9"></span>EMISSIONS COMPARISON</div>
        </div>""", unsafe_allow_html=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=["Immediate", "Optimised"],
            y=[worst_em, best_em],
            text=[f"{worst_em:.4f}g", f"{best_em:.4f}g"],
            textposition="auto",
            textfont=dict(family="IBM Plex Mono", size=11, color="#e2e8f0"),
            marker_color=["#ef4444", "#22c55e"],
            marker_line_width=0, width=0.5
        ))
        fig2.update_layout(
            height=290, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=10, b=10),
            xaxis=dict(showgrid=False, color="#334155",
                       tickfont=dict(family="IBM Plex Mono", size=11)),
            yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
                       tickfont=dict(family="IBM Plex Mono", size=10), title="g CO₂"),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SYSTEM LOG
    import datetime
    ts = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
    best_start_str  = best['start'].strftime('%Y-%m-%d %H:%M') if hasattr(best['start'], 'strftime') else str(best['start'])
    worst_start_str = worst['start'].strftime('%Y-%m-%d %H:%M') if hasattr(worst['start'], 'strftime') else str(worst['start'])
    st.markdown(f"""
    <div class="sys-log">
        <span class="log-ts">[{ts}]</span><span class="log-g">INIT</span>  National Grid ESO forecast loaded · {len(df)} intervals · 30-min resolution<br>
        <span class="log-ts">[{ts}]</span><span class="log-g">SCHED</span> Strategy selected: <strong style="color:#94a3b8">{strategy.upper()}</strong> · grid volatility σ = {float(df['carbon'].std()):.1f} gCO₂/kWh<br>
        <span class="log-ts">[{ts}]</span><span class="log-g">OPT</span>   Best window → {best_start_str} · avg <strong style="color:#22c55e">{best['avg_carbon']:.1f} gCO₂/kWh</strong><br>
        <span class="log-ts">[{ts}]</span><span class="log-g">BASE</span>  Baseline window → {worst_start_str} · avg <strong style="color:#ef4444">{worst['avg_carbon']:.1f} gCO₂/kWh</strong><br>
        <span class="log-ok">✓ DONE</span> Optimisation complete · <strong style="color:#22c55e">{savings:.1f}% CO₂ reduction</strong> achieved by delaying {delay_h:.1f} h
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── PDF REPORT
    r1, _ = st.columns([1, 4])
    with r1:
        if st.button("⬇  Generate PDF Report", type="secondary"):
            file = generate_report(savings, best, worst)
            with open(file, "rb") as f:
                st.download_button("Download Report", f, file_name="caml_tc_report.pdf",
                                   mime="application/pdf")
else:
    # ── EMPTY STATE
    st.markdown("""
    <div style="
        background: #0b1520;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 14px;
        padding: 80px 40px;
        text-align: center;
        margin-top: 12px;
    ">
        <div style="
            font-family: 'IBM Plex Mono', monospace;
            font-size: 32px; color: #22c55e; opacity: 0.25;
            margin-bottom: 20px; letter-spacing: -0.02em;
        ">⚡</div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#1e293b; letter-spacing:0.05em; text-transform:uppercase;">
            Configure training duration and urgency · then run optimisation
        </div>
        <div style="font-size:12px; color:#1e293b; margin-top:10px;">
            The scheduler will query the National Grid ESO API and return the optimal execution window
        </div>
    </div>
    """, unsafe_allow_html=True)