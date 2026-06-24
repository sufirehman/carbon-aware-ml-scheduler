import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from camltc.carbon_api import CarbonAPI

st.set_page_config(
    page_title="CAML-TC · Forecast",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

.page-hdr { margin-bottom: 28px; }
.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #0ea5e9;
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.2);
    padding: 4px 14px; border-radius: 6px;
    letter-spacing: 0.1em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-hdr h1 { font-size: 26px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0 0 6px; }
.page-hdr p  { font-size: 13px; color: #475569; margin: 0; }

/* ── STAT CARDS ── */
.stat { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-left: 2px solid transparent; border-radius: 10px; padding: 18px 20px; }
.stat-r { border-left-color: #ef4444 !important; }
.stat-g { border-left-color: #22c55e !important; }
.stat-b { border-left-color: #0ea5e9 !important; }
.stat-a { border-left-color: #f59e0b !important; }
.stat-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.stat-val { font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:600; color:#f1f5f9; letter-spacing:-0.02em; margin-bottom:4px; }
.stat-sub { font-size:11px; color:#475569; }

/* ── CHART CARD ── */
.chart-card  { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.chart-hdr   { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.chart-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; display:flex; align-items:center; gap:8px; }
.chart-meta  { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#1e293b; }
.chart-sub   { font-size: 12px; color: #94a3b8; margin-bottom: 14px; }
.cdot        { width:7px; height:7px; border-radius:50%; display:inline-block; flex-shrink:0; }

/* ── WINDOW LIST ── */
.win-item {
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.win-item:last-child { border-bottom: none; }
.win-time { font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:500; }
.win-avg  { font-size:11px; color:#475569; margin-top:3px; }
.win-badge {
    font-family:'IBM Plex Mono',monospace; font-size:10px;
    padding: 4px 12px; border-radius: 4px;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.b-best { color:#22c55e; background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); }
.b-good { color:#f59e0b; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); }
.b-ok   { color:#475569; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); }

/* ── INSIGHT ── */
.insight {
    background: rgba(34,197,94,0.03);
    border: 1px solid rgba(34,197,94,0.1);
    border-radius: 8px; padding: 20px;
    font-size: 13px; color: #64748b; line-height: 1.8;
}
.ins-lbl {
    font-family:'IBM Plex Mono',monospace;
    font-size:10px; color:#22c55e;
    text-transform:uppercase; letter-spacing:0.12em;
    margin-bottom:10px;
}

/* ── TABS ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1520 !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11px !important; color: #334155 !important;
    letter-spacing: 0.07em !important; text-transform: uppercase !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #22c55e !important;
    border-bottom: 2px solid #22c55e !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: #0b1520 !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
    padding: 20px !important;
}

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #08111c !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #64748b !important; }

/* Plotly */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
/* Dataframe */
div[data-testid="stDataFrame"] { background: transparent !important; }

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── HEADER
st.markdown("""
<div class="page-hdr">
    <div class="page-badge">Forecast</div>
    <h1>Carbon Intensity Forecast System</h1>
    <p>Real-time 24-hour UK grid carbon prediction · National Grid ESO · 30-min resolution · uncertainty-weighted</p>
</div>
""", unsafe_allow_html=True)

# ── LOAD DATA
with st.spinner("Querying National Grid ESO..."):
    api = CarbonAPI()
    df = api.get_24h_forecast()
    df["carbon"] = df["actual"].fillna(df["forecast"])
    df["from"] = pd.to_datetime(df["from"])
    df["confidence"] = df["actual"].notna().map({True: "Actual", False: "Forecast"})

# ── ANALYTICS
peak      = df["carbon"].max()
low       = df["carbon"].min()
avg       = df["carbon"].mean()
vol       = df["carbon"].std()
peak_time = df.loc[df["carbon"].idxmax(), "from"]
low_time  = df.loc[df["carbon"].idxmin(), "from"]
swing_pct = (peak - low) / peak * 100

# derive real scheduling windows from data
df_sorted = df.nsmallest(6, "carbon")
window_size = 3  # 3 × 30-min = 1.5 hr windows

# find the top 3 non-overlapping windows of 3 consecutive slots
def find_windows(df, n=3, size=3):
    windows = []
    used = set()
    sorted_idx = df["carbon"].argsort().values
    for i in sorted_idx:
        if i in used:
            continue
        # check if we can form a window here
        if i + size <= len(df):
            indices = list(range(i, i + size))
            if not any(j in used for j in indices):
                w_df = df.iloc[indices]
                windows.append({
                    "start": w_df.iloc[0]["from"],
                    "end": w_df.iloc[-1]["from"] + pd.Timedelta(minutes=30),
                    "avg": w_df["carbon"].mean(),
                    "duration_h": size * 0.5,
                })
                for j in indices:
                    used.add(j)
        if len(windows) == n:
            break
    return sorted(windows, key=lambda x: x["avg"])

windows = find_windows(df, n=3, size=3)

# ── STAT CARDS
s1, s2, s3, s4 = st.columns(4)
cards = [
    (s1, "stat stat-r", "Peak Carbon",      f"{peak:.0f}",  f"gCO₂/kWh · {peak_time.strftime('%H:%M')}"),
    (s2, "stat stat-g", "Lowest Carbon",    f"{low:.0f}",   f"gCO₂/kWh · {low_time.strftime('%H:%M')}"),
    (s3, "stat stat-b", "24-hr Average",    f"{avg:.0f}",   "gCO₂/kWh"),
    (s4, "stat stat-a", "Grid Volatility",  f"±{vol:.0f}",  f"σ · {swing_pct:.0f}% peak-to-low swing"),
]
for col, cls, lbl, val, sub in cards:
    with col:
        st.markdown(f"""<div class="{cls}">
            <div class="stat-lbl">{lbl}</div>
            <div class="stat-val">{val}</div>
            <div class="stat-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── MAIN CHART
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown("""
<div class="chart-hdr">
    <div class="chart-title">
        <span class="cdot" style="background:#22c55e"></span>Actual
        <span class="cdot" style="background:#0ea5e9; margin-left:6px;"></span>Forecast
        &nbsp;&middot;&nbsp; 24-Hour Carbon Intensity Curve
    </div>
    <div class="chart-meta">gCO&#8322;/kWh &middot; hover for values &middot; optimal window shaded</div>
</div>
""", unsafe_allow_html=True)

fig = go.Figure()

# Colour-encoded line by carbon level
colors = df["carbon"].apply(
    lambda v: "#22c55e" if v <= low + (peak - low) * 0.33
    else ("#f59e0b" if v <= low + (peak - low) * 0.66 else "#ef4444")
)

# Area base
fig.add_trace(go.Scatter(
    x=df["from"], y=df["carbon"],
    mode="lines", name="Carbon Intensity",
    line=dict(color="#475569", width=2),
    fill="tozeroy", fillcolor="rgba(71,85,105,0.04)",
    hovertemplate="<b>%{x|%H:%M}</b><br>%{y:.1f} gCO₂/kWh<extra></extra>",
    showlegend=False
))

# Actual vs Forecast distinction
actual_df   = df[df["confidence"] == "Actual"]
forecast_df = df[df["confidence"] == "Forecast"]

if len(actual_df):
    fig.add_trace(go.Scatter(
        x=actual_df["from"], y=actual_df["carbon"],
        mode="lines", name="Actual",
        line=dict(color="#22c55e", width=2.5),
        hovertemplate="<b>%{x|%H:%M}</b><br>Actual: %{y:.1f} gCO₂/kWh<extra></extra>"
    ))
if len(forecast_df):
    fig.add_trace(go.Scatter(
        x=forecast_df["from"], y=forecast_df["carbon"],
        mode="lines", name="Forecast",
        line=dict(color="#0ea5e9", width=2, dash="dot"),
        hovertemplate="<b>%{x|%H:%M}</b><br>Forecast: %{y:.1f} gCO₂/kWh<extra></extra>"
    ))

# Highlight best window
if windows:
    fig.add_vrect(
        x0=windows[0]["start"], x1=windows[0]["end"],
        fillcolor="rgba(34,197,94,0.1)", line_color="rgba(34,197,94,0.3)", line_width=1,
        annotation_text="Best Window",
        annotation_font_color="#22c55e", annotation_font_size=11,
        annotation_font_family="IBM Plex Mono"
    )

# Peak marker
fig.add_trace(go.Scatter(
    x=[peak_time], y=[peak], mode="markers+text",
    marker=dict(size=10, color="#ef4444", symbol="circle-open", line_width=2),
    text=[f"PEAK {peak:.0f}"],
    textfont=dict(family="IBM Plex Mono", size=10, color="#ef4444"),
    textposition="top right", showlegend=False
))

# Low marker
fig.add_trace(go.Scatter(
    x=[low_time], y=[low], mode="markers+text",
    marker=dict(size=10, color="#22c55e", symbol="circle-open", line_width=2),
    text=[f"LOW {low:.0f}"],
    textfont=dict(family="IBM Plex Mono", size=10, color="#22c55e"),
    textposition="bottom right", showlegend=False
))

fig.update_layout(
    height=380, template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=10, t=10, b=10),
    font=dict(family="IBM Plex Mono", size=10, color="#475569"),
    xaxis=dict(showgrid=False, color="#334155",
               tickfont=dict(family="IBM Plex Mono", size=10)),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
               tickfont=dict(family="IBM Plex Mono", size=10), title="gCO₂/kWh"),
    hovermode="x unified",
    legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#475569"),
                bgcolor="rgba(0,0,0,0)")
)
st.plotly_chart(fig, use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# ── LOWER PANELS
col_l, col_r = st.columns(2)

with col_l:
    badge_classes = ["b-best", "b-good", "b-ok"]
    badge_labels  = ["BEST", "GOOD", "OK"]

    win_items = []
    for i, w in enumerate(windows):
        bc     = badge_classes[i] if i < len(badge_classes) else "b-ok"
        bl     = badge_labels[i]  if i < len(badge_labels)  else "OK"
        color  = "#22c55e" if i == 0 else ("#f59e0b" if i == 1 else "#475569")
        saving = ((avg - w["avg"]) / avg * 100) if avg > 0 else 0
        t_start = w["start"].strftime("%H:%M")
        t_end   = w["end"].strftime("%H:%M")
        avg_ci  = f"{w['avg']:.0f}"
        dur     = f"{w['duration_h']:.1f}"
        sav_pct = f"{max(saving, 0):.0f}"
        win_items.append(
            '<div class="win-item">'
            '<div>'
            f'<div class="win-time" style="color:{color}">{t_start} &ndash; {t_end}</div>'
            f'<div class="win-avg">Avg {avg_ci} gCO&#8322;/kWh &middot; {dur}h window &middot; ~{sav_pct}% below average</div>'
            '</div>'
            f'<div class="win-badge {bc}">{bl}</div>'
            '</div>'
        )

    card_html = (
        '<div class="chart-card">'
        '<div class="chart-hdr">'
        '<div class="chart-title"><span class="cdot" style="background:#22c55e"></span>Optimal Scheduling Windows</div>'
        '<div class="chart-meta">from live forecast data</div>'
        '</div>'
        + "".join(win_items) +
        '</div>'
    )
    st.markdown(card_html, unsafe_allow_html=True)

with col_r:
    max_saving   = min(swing_pct, 70)
    vol_str      = f"{vol:.0f}"
    swing_str    = f"{swing_pct:.0f}"
    low_time_str = low_time.strftime("%H:%M")
    max_sav_str  = f"{max_saving:.0f}"
    strategy_str = "RL agent engaged &mdash; high grid volatility detected." if vol > 30 else "Heuristic scheduler sufficient &mdash; moderate grid volatility."

    insight_html = (
        '<div class="chart-card">'
        '<div class="chart-hdr">'
        '<div class="chart-title"><span class="cdot" style="background:#22c55e"></span>Grid Intelligence Analysis</div>'
        '<div class="chart-meta">carbon opportunity score</div>'
        '</div>'
        '<div class="insight">'
        '<div class="ins-lbl">Today&#39;s Carbon Opportunity</div>'
        f'The UK grid shows <strong style="color:#94a3b8">&plusmn;{vol_str} gCO&#8322;/kWh</strong> volatility '
        f'today &mdash; a <strong style="color:#22c55e">{swing_str}%</strong> peak-to-low swing '
        'across 24 hours. This is the window of opportunity CAML-TC exploits.<br><br>'
        f'<strong style="color:#94a3b8">Recommended action:</strong> Schedule GPU training '
        f'starting <strong style="color:#22c55e">{low_time_str}</strong> for '
        'maximum carbon efficiency. Expected saving vs immediate execution: '
        f'up to <strong style="color:#22c55e">{max_sav_str}%</strong>.<br><br>'
        f'<strong style="color:#94a3b8">Strategy:</strong> {strategy_str}'
        '</div></div>'
    )
    st.markdown(insight_html, unsafe_allow_html=True)

# ── DATA TABS
st.markdown("<br>", unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Summary", "Raw Forecast Data", "Research Notes"])

with tab1:
    col_a, col_b = st.columns(2)
    with col_a:
        st.json({
            "peak_gCO2_kWh":      round(float(peak), 1),
            "low_gCO2_kWh":       round(float(low), 1),
            "average_gCO2_kWh":   round(float(avg), 1),
            "volatility_std":     round(float(vol), 1),
            "peak_time_utc":      str(peak_time),
            "low_time_utc":       str(low_time),
            "data_intervals":     len(df),
            "scheduling_opportunity_pct": round(float(swing_pct), 1),
        })
    with col_b:
        st.json({
            "best_window_start":  str(windows[0]["start"]) if windows else "N/A",
            "best_window_avg":    round(windows[0]["avg"], 1) if windows else "N/A",
            "strategy_trigger":   "rl" if vol > 30 else "heuristic",
            "actual_points":      int(df["confidence"].eq("Actual").sum()),
            "forecast_points":    int(df["confidence"].eq("Forecast").sum()),
        })

with tab2:
    display = df[["from", "forecast", "actual", "carbon", "confidence"]].copy()
    display.columns = ["Timestamp", "Forecast (gCO₂/kWh)", "Actual (gCO₂/kWh)", "Used (gCO₂/kWh)", "Source"]
    st.dataframe(display, use_container_width=True, height=300)

with tab3:
    st.markdown(
        '<div class="insight" style="margin-top:4px;">'
        '<div class="ins-lbl">Why temporal carbon variation enables carbon-aware scheduling</div>'
        'The UK grid&#39;s carbon intensity is highly time-dependent, driven by renewable intermittency '
        '(wind and solar), demand peaks (morning and evening), and baseload mix. '
        'CAML-TC exploits three temporal properties:<br><br>'
        '<strong style="color:#94a3b8">1. Diurnal patterns</strong> &mdash; low demand overnight, predictable low-carbon windows<br>'
        '<strong style="color:#94a3b8">2. Renewable stochasticity</strong> &mdash; wind bursts create short-lived green windows the RL agent learns to identify<br>'
        '<strong style="color:#94a3b8">3. Forecast reliability decay</strong> &mdash; confidence drops with horizon; '
        'CAML-TC weights near-term slots higher via exponential decay <code>w&#8345; = exp(&minus;&lambda;&middot;&Delta;t)</code><br><br>'
        'The larger the daily swing, the greater the potential saving. On high-variability days (&sigma; &gt; 30), '
        'the RL agent activates in place of the heuristic &mdash; this is when adaptive learning has the most to offer.'
        '</div>',
        unsafe_allow_html=True
    )