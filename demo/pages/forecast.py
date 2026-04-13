import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from core.carbon_api import CarbonAPI

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Carbon Intelligence Forecast",
    layout="wide"
)

# ----------------------------
# GLOBAL STYLING
# ----------------------------
st.markdown("""
<style>

body {
    background-color: #0f172a;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 20px rgba(0,0,0,0.2);
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 28px;
    font-weight: 700;
    color: white;
    margin-top: 6px;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-top: 20px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:48px; background:linear-gradient(90deg,#00C9FF,#92FE9D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;">
📡 Carbon Intelligence Forecast System
</h1>
<p style="color:#94a3b8; font-size:18px;">
Real-time UK grid carbon prediction for ML scheduling optimization
</p>
</div>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
api = CarbonAPI()
df = api.get_24h_forecast()

df["carbon"] = df["actual"].fillna(df["forecast"])
df["from"] = pd.to_datetime(df["from"])

# ----------------------------
# ANALYTICS
# ----------------------------
peak = df["carbon"].max()
low = df["carbon"].min()
avg = df["carbon"].mean()
volatility = df["carbon"].std()

peak_time = df.loc[df["carbon"].idxmax(), "from"]
low_time = df.loc[df["carbon"].idxmin(), "from"]

best_window = df.nsmallest(3, "carbon")
worst_window = df.nlargest(3, "carbon")

# ----------------------------
# KPI CARDS (FIXED - NO RAW HTML ISSUE)
# ----------------------------
st.markdown("## 📊 Grid Intelligence Overview")

st.markdown(f"""
<div class="card-grid">

    <div class="metric-card">
        <div class="metric-title">🔥 Peak Carbon</div>
        <div class="metric-value">{peak:.2f}</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">🌱 Lowest Carbon</div>
        <div class="metric-value">{low:.2f}</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">📊 Average</div>
        <div class="metric-value">{avg:.2f}</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">📉 Volatility</div>
        <div class="metric-value">{volatility:.2f}</div>
    </div>

</div>
""", unsafe_allow_html=True)

# ----------------------------
# INSIGHT SECTION
# ----------------------------
st.markdown("## 🧠 AI Scheduling Recommendation")

st.success(f"""
✔ Optimal Low-Carbon Window Identified

Start Time: **{best_window.iloc[0]['from'].strftime('%H:%M')}**

Why this window:
- Lowest carbon cluster detected
- Stable grid conditions
- Best for ML training execution
- Ideal RL scheduling target

Expected CO₂ savings vs peak: **High**
""")

# ----------------------------
# GRAPH (WITH RED + GREEN WINDOWS FIXED)
# ----------------------------
st.markdown("## 📈 24-Hour Carbon Intelligence Curve")

fig = go.Figure()

# main line
fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    name="Carbon Intensity",
    line=dict(width=3, color="#111827")
))

# 🟢 BEST WINDOW
fig.add_vrect(
    x0=best_window.iloc[0]["from"],
    x1=best_window.iloc[-1]["from"],
    fillcolor="green",
    opacity=0.12,
    line_width=0,
    annotation_text="Low Carbon Zone"
)

# 🔴 WORST WINDOW
fig.add_vrect(
    x0=worst_window.iloc[0]["from"],
    x1=worst_window.iloc[-1]["from"],
    fillcolor="red",
    opacity=0.12,
    line_width=0,
    annotation_text="High Carbon Zone"
)

# peak
fig.add_trace(go.Scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center"
))

# low
fig.add_trace(go.Scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center"
))

fig.update_layout(
    height=600,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Carbon Intensity (gCO₂/kWh)"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# DATA PANEL (NO BUTTONS)
# ----------------------------
st.markdown("## 📂 Forecast Intelligence Panel")

with st.expander("📊 View Raw Forecast Data (Click to Expand)"):
    st.dataframe(df[["from", "carbon"]], use_container_width=True)

with st.expander("🧠 System Insights"):
    st.info("""
The UK grid shows strong temporal variation in carbon intensity.

This enables:
- Carbon-aware ML scheduling
- RL-based optimization
- Dynamic workload shifting
""")