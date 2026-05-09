import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from core.carbon_api import CarbonAPI

st.set_page_config(
    page_title="CarbonML · Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL UI THEME (MATCH APP.PY)
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
    padding: 2rem 2.5rem;
    max-width: 100%;
}

/* grid background */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* HEADER */
.header {
    margin-bottom: 25px;
}

.badge {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    padding: 6px 14px;
    border-radius: 20px;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    letter-spacing: 0.08em;
}

h1 {
    font-size: 30px;
    margin-top: 10px;
    margin-bottom: 6px;
    color: #f8fafc;
}

.sub {
    color: #94a3b8;
    font-size: 13px;
}

/* CARD */
.card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 18px;
}

/* STAT */
.stat-label {
    font-family: 'IBM Plex Mono';
    font-size: 10px;
    color: #64748b;
    text-transform: uppercase;
}

.stat-value {
    font-family: 'IBM Plex Mono';
    font-size: 24px;
    font-weight: 600;
    color: #f8fafc;
}

.stat-unit {
    font-size: 11px;
    color: #94a3b8;
}

/* WINDOW */
.window {
    display: flex;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}

.window:last-child { border: none; }

.window-time {
    font-family: 'IBM Plex Mono';
    font-size: 13px;
    color: #f8fafc;
}

.badge-best {
    color: #22c55e;
}

.badge-good {
    color: #f59e0b;
}

/* INSIGHT */
.insight {
    background: rgba(34,197,94,0.04);
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 10px;
    padding: 16px;
    font-size: 13px;
    color: #94a3b8;
    line-height: 1.7;
}

.insight-title {
    font-family: 'IBM Plex Mono';
    font-size: 10px;
    color: #22c55e;
    text-transform: uppercase;
    margin-bottom: 8px;
}

/* TAB CLEANUP */
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono';
    font-size: 11px;
    color: #64748b;
}

.stTabs [aria-selected="true"] {
    color: #22c55e !important;
}

</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="header">
    <div class="badge">FORECAST ENGINE</div>
    <h1>Carbon Intelligence Forecast</h1>
    <div class="sub">
        UK National Grid real-time carbon prediction · 30-min resolution · ML scheduling optimization
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# DATA
# =========================
with st.spinner("Loading carbon forecast..."):
    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])
    df["from"] = pd.to_datetime(df["from"])

# =========================
# METRICS
# =========================
peak = df["carbon"].max()
low = df["carbon"].min()
avg = df["carbon"].mean()
std = df["carbon"].std()

peak_time = df.loc[df["carbon"].idxmax(), "from"]
low_time = df.loc[df["carbon"].idxmin(), "from"]

m1, m2, m3, m4 = st.columns(4)

metrics = [
    ("Peak", f"{peak:.0f}", "gCO₂/kWh"),
    ("Lowest", f"{low:.0f}", "gCO₂/kWh"),
    ("Average", f"{avg:.0f}", "gCO₂/kWh"),
    ("Volatility", f"±{std:.0f}", "std dev"),
]

for col, (label, val, unit) in zip([m1, m2, m3, m4], metrics):
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="stat-label">{label}</div>
            <div class="stat-value">{val}</div>
            <div class="stat-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# =========================
# CHART
# =========================
st.markdown("""
<div class="card">
    <div class="stat-label">24H CARBON INTENSITY</div>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    line=dict(width=2),
    fill="tozeroy",
    fillcolor="rgba(100,116,139,0.06)"
))

fig.update_layout(
    height=380,
    template="plotly_dark",
    margin=dict(l=10, r=10, t=10, b=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# WINDOWS + INSIGHT
# =========================
c1, c2 = st.columns(2)

with c1:
    st.markdown("""
    <div class="card">
        <div class="stat-label">OPTIMAL WINDOWS</div>
    """, unsafe_allow_html=True)

    best = df.nsmallest(3, "carbon")

    for _, r in best.iterrows():
        st.markdown(f"""
        <div class="window">
            <div class="window-time">{r['from'].strftime('%H:%M')}</div>
            <div class="badge-best">LOW</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="card">
        <div class="stat-label">SYSTEM INSIGHT</div>
        <div class="insight">
            <div class="insight-title">ANALYSIS</div>
            Carbon intensity ranges from <b>{low:.0f}</b> to <b>{peak:.0f}</b> gCO₂/kWh.<br><br>
            Scheduling ML workloads during low-carbon windows can reduce emissions
            by up to <b style="color:#22c55e">{((peak-low)/peak*100):.0f}%</b>.<br><br>
            Recommended execution window aligns with <b>{low_time.strftime('%H:%M')}</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# DATA
# =========================
tab1, tab2 = st.tabs(["Summary", "Raw Data"])

with tab1:
    st.json({
        "peak": float(peak),
        "low": float(low),
        "avg": float(avg),
        "std": float(std),
        "peak_time": str(peak_time),
        "low_time": str(low_time)
    })

with tab2:
    st.dataframe(df[["from", "carbon"]], use_container_width=True)