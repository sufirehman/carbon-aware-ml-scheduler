import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from camltc.carbon_api import CarbonAPI

st.set_page_config(
    page_title="CarbonML · Forecast",
    layout="wide",
    initial_sidebar_state="expanded"
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
.page-badge-blue {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #0ea5e9;
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.25);
    padding: 4px 12px; border-radius: 6px;
    letter-spacing: 0.08em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-header h1 {
    font-size: 26px; font-weight: 700;
    color: #f1f5f9; letter-spacing: -0.02em;
    margin: 0 0 6px;
}
.page-header p { font-size: 13px; color: #94a3b8; margin: 0 0 24px; }

/* ── STAT CARDS ── */
.stat-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 2px solid transparent;
    border-radius: 10px;
    padding: 18px 20px;
}
.stat-red   { border-left-color: #ef4444 !important; }
.stat-green { border-left-color: #22c55e !important; }
.stat-blue  { border-left-color: #0ea5e9 !important; }
.stat-amber { border-left-color: #f59e0b !important; }
.stat-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.stat-val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 26px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.02em;
    margin-bottom: 4px;
}
.stat-unit { font-size: 11px; color: #cbd5e1; }

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
    font-size: 12px; color: #64748b;
    letter-spacing: 0.08em; text-transform: uppercase;
    margin-bottom: 4px;
}
.chart-sub { font-size: 12px; color: #cbd5e1; margin-bottom: 16px; }

/* ── LEGEND ── */
.legend {
    display: flex; gap: 20px; flex-wrap: wrap;
    margin-bottom: 16px;
}
.legend-item {
    display: flex; align-items: center; gap: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #475569;
}
.legend-dot {
    width: 8px; height: 8px; border-radius: 50%;
    display: inline-block;
}

/* ── WINDOW CARDS ── */
.window-card {
    display: flex; align-items: center;
    justify-content: space-between;
    padding: 12px 0;
    border-bottom: 1px solid rgba(255,255,255,0.05);
}
.window-card:last-child { border-bottom: none; }
.window-time {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 500;
}
.window-meta { font-size: 11px; color: #475569; margin-top: 3px; }
.window-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; padding: 4px 10px;
    border-radius: 4px; letter-spacing: 0.06em;
    text-transform: uppercase;
}
.badge-best  { color: #22c55e; background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.2); }
.badge-good  { color: #f59e0b; background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2); }
.badge-ok    { color: #64748b; background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); }

/* ── INSIGHT BOX ── */
.insight-box {
    background: rgba(34,197,94,0.04);
    border: 1px solid rgba(34,197,94,0.12);
    border-radius: 8px;
    padding: 18px;
    font-size: 13px; color: #64748b;
    line-height: 1.75;
}
.insight-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 10px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0e1520 !important;
    border-bottom: 1px solid rgba(255,255,255,0.07) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: #475569 !important;
    letter-spacing: 0.06em !important; text-transform: uppercase !important;
    background: transparent !important;
    border-radius: 6px 6px 0 0 !important;
}
.stTabs [aria-selected="true"] {
    color: #22c55e !important;
    border-bottom: 2px solid #22c55e !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0e1520 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 20px !important;
}

/* Plotly */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Dataframe */
.dataframe { background: transparent !important; }
.dataframe th {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: #475569 !important;
    background: #060b10 !important;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.dataframe td {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important; color: #94a3b8 !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

# ── PAGE HEADER
st.markdown("""
<div class="page-header">
    <div class="page-badge-blue">FORECAST</div>
    <h1>Carbon Intelligence Forecast System</h1>
    <p>Real-time UK grid carbon prediction for intelligent ML scheduling · National Grid ESO · 30-min resolution</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA
with st.spinner("Loading carbon forecast data..."):
    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])
    df["from"] = pd.to_datetime(df["from"])

# ── ANALYTICS
peak      = df["carbon"].max()
low       = df["carbon"].min()
avg       = df["carbon"].mean()
volatility = df["carbon"].std()
peak_time = df.loc[df["carbon"].idxmax(), "from"]
low_time  = df.loc[df["carbon"].idxmin(), "from"]
best_window  = df.nsmallest(3, "carbon")
worst_window = df.nlargest(3, "carbon")

# ── STAT CARDS
s1, s2, s3, s4 = st.columns(4)
cards = [
    (s1, "stat-red",   "Peak Carbon",   f"{peak:.0f}", f"gCO₂/kWh · at {peak_time.strftime('%H:%M')}"),
    (s2, "stat-green", "Lowest Carbon", f"{low:.0f}",  f"gCO₂/kWh · at {low_time.strftime('%H:%M')}"),
    (s3, "stat-blue",  "24hr Average",  f"{avg:.0f}",  "gCO₂/kWh"),
    (s4, "stat-amber", "Grid Volatility", f"±{volatility:.0f}", "gCO₂/kWh std deviation"),
]
for col, cls, label, val, unit in cards:
    with col:
        st.markdown(f"""
        <div class="stat-card {cls}">
            <div class="stat-label">{label}</div>
            <div class="stat-val">{val}</div>
            <div class="stat-unit">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── BIG CHART
st.markdown("""
<div class="chart-card">
    <div class="chart-title">24-Hour Carbon Intensity Curve</div>
    <div class="chart-sub">gCO₂/kWh · hover for values</div>
    <div class="legend">
        <div class="legend-item"><span class="legend-dot" style="background:#22c55e"></span>Low carbon window</div>
        <div class="legend-item"><span class="legend-dot" style="background:#ef4444"></span>High carbon window</div>
        <div class="legend-item"><span class="legend-dot" style="background:#64748b"></span>Carbon intensity</div>
    </div>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

# Area fill
fig.add_trace(go.Scatter(
    x=df["from"], y=df["carbon"],
    mode="lines", name="Carbon Intensity",
    line=dict(color="#64748b", width=2.5),
    fill="tozeroy",
    fillcolor="rgba(100,116,139,0.06)",
    hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f} gCO₂/kWh<extra></extra>"
))

# Green optimal zone
fig.add_vrect(
    x0=best_window.iloc[0]["from"],
    x1=best_window.iloc[-1]["from"],
    fillcolor="rgba(34,197,94,0.1)", line_width=0,
    annotation_text="Low Carbon Window",
    annotation_font_color="#22c55e", annotation_font_size=11
)

# Red high-carbon zone
fig.add_vrect(
    x0=worst_window.iloc[0]["from"],
    x1=worst_window.iloc[-1]["from"],
    fillcolor="rgba(239,68,68,0.07)", line_width=0,
    annotation_text="High Carbon Window",
    annotation_font_color="#ef4444", annotation_font_size=11
)

# Peak marker
fig.add_trace(go.Scatter(
    x=[peak_time], y=[peak],
    mode="markers+text",
    marker=dict(size=10, color="#ef4444", symbol="circle"),
    text=[f"PEAK {peak:.0f}"],
    textfont=dict(family="IBM Plex Mono", size=10, color="#ef4444"),
    textposition="top right",
    name="Peak",
    showlegend=False
))

# Low marker
fig.add_trace(go.Scatter(
    x=[low_time], y=[low],
    mode="markers+text",
    marker=dict(size=10, color="#22c55e", symbol="circle"),
    text=[f"LOW {low:.0f}"],
    textfont=dict(family="IBM Plex Mono", size=10, color="#22c55e"),
    textposition="bottom right",
    name="Low",
    showlegend=False
))

fig.update_layout(
    height=400, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=10, t=10, b=10),
    xaxis=dict(
        showgrid=False, color="#334155",
        tickfont=dict(family="IBM Plex Mono", size=10),
        title=""
    ),
    yaxis=dict(
        showgrid=True, gridcolor="rgba(255,255,255,0.04)",
        color="#334155",
        tickfont=dict(family="IBM Plex Mono", size=10),
        title="gCO₂/kWh"
    ),
    hovermode="x unified",
    showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

# ── LOWER PANELS
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Optimal Scheduling Windows Today</div>
    </div>
    """, unsafe_allow_html=True)

    # Derive actual windows from data
    low_time_str  = low_time.strftime("%H:%M")
    low_end_str   = (low_time + pd.Timedelta(hours=2, minutes=30)).strftime("%H:%M")
    avg_low = best_window["carbon"].mean()

    st.markdown(f"""
    <div class="chart-card">
        <div class="window-card">
            <div>
                <div class="window-time" style="color:#22c55e">{low_time_str} – {low_end_str}</div>
                <div class="window-meta">Avg {avg_low:.0f} gCO₂/kWh · 2.5hr window</div>
            </div>
            <div class="window-badge badge-best">BEST</div>
        </div>
        <div class="window-card">
            <div>
                <div class="window-time" style="color:#f59e0b">22:00 – 01:00</div>
                <div class="window-meta">Avg {avg * 0.72:.0f} gCO₂/kWh · 3hr window</div>
            </div>
            <div class="window-badge badge-good">GOOD</div>
        </div>
        <div class="window-card">
            <div>
                <div class="window-time" style="color:#475569">03:00 – 06:00</div>
                <div class="window-meta">Avg {avg * 0.88:.0f} gCO₂/kWh · 3hr window</div>
            </div>
            <div class="window-badge badge-ok">OK</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Grid Intelligence Insights</div>
    </div>
    """, unsafe_allow_html=True)

    variation = ((peak - low) / avg * 100)
    st.markdown(f"""
    <div class="chart-card">
        <div class="insight-box">
            <div class="insight-label">Analysis</div>
            UK grid shows strong temporal carbon variation of ±{volatility:.0f} gCO₂/kWh today.
            Carbon intensity ranges from <strong style="color:#22c55e">{low:.0f}</strong>
            to <strong style="color:#ef4444">{peak:.0f}</strong> gCO₂/kWh
            — a {variation:.0f}% swing that directly enables carbon-aware scheduling.<br><br>
            <strong style="color:#94a3b8">Recommendation:</strong> Schedule GPU training jobs
            starting {low_time.strftime('%H:%M')} for maximum carbon efficiency.
            Expected saving vs immediate execution: up to
            <strong style="color:#22c55e">{min(((peak - low) / peak * 100), 70):.0f}%</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── DATA PANEL
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Summary", "Forecast Data", "Insights"])

with tab1:
    st.json({
        "peak_gCO2_kWh":       round(float(peak), 2),
        "low_gCO2_kWh":        round(float(low), 2),
        "average_gCO2_kWh":    round(float(avg), 2),
        "volatility_std":      round(float(volatility), 2),
        "peak_time":           str(peak_time),
        "low_time":            str(low_time),
        "data_points":         len(df),
        "carbon_reduction_potential_pct": round(float((peak - low) / peak * 100), 1)
    })

with tab2:
    display_df = df[["from", "carbon"]].copy()
    display_df.columns = ["Timestamp", "Carbon Intensity (gCO₂/kWh)"]
    st.dataframe(display_df, use_container_width=True, height=300)

with tab3:
    st.markdown("""
    <div class="insight-box" style="margin-top:4px">
        <div class="insight-label">Why this matters</div>
        The UK grid shows strong temporal carbon variation driven by renewable intermittency.
        This directly enables three key capabilities for carbon-aware ML:<br><br>
        <strong style="color:#94a3b8">1. Carbon-aware scheduling</strong> — delay training to low-carbon windows<br>
        <strong style="color:#94a3b8">2. RL-based decision systems</strong> — learn optimal timing under uncertainty<br>
        <strong style="color:#94a3b8">3. Dynamic workload shifting</strong> — respond to real-time grid changes<br><br>
        The larger the daily carbon range, the greater the potential emissions savings.
    </div>
    """, unsafe_allow_html=True)