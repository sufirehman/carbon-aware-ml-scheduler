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
# HERO SECTION
# ----------------------------
st.markdown("""
<div style="text-align:center;">
<h1 style="font-size:48px; background:linear-gradient(90deg,#00C9FF,#92FE9D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;">
📡 Carbon Intelligence Forecast System
</h1>
<p style="color:#94a3b8; font-size:18px;">
Real-time UK grid carbon prediction for intelligent ML scheduling
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
# KPI SECTION (UPGRADED STYLE)
# ----------------------------
st.markdown("## 📊 Grid Intelligence Overview")

st.markdown("""
<style>
.card-container {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
    margin-top: 10px;
}

.metric-card {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    padding: 18px;
    border-radius: 16px;
    text-align: center;
    backdrop-filter: blur(10px);
}

.metric-title {
    font-size: 14px;
    color: #94a3b8;
}

.metric-value {
    font-size: 26px;
    font-weight: bold;
    color: white;
    margin-top: 6px;
}
</style>

<div class="card-container">

    <div class="metric-card">
        <div class="metric-title">🔥 Peak Carbon</div>
        <div class="metric-value">""" + f"{peak:.2f}" + """</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">🌱 Lowest Carbon</div>
        <div class="metric-value">""" + f"{low:.2f}" + """</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">📊 Average</div>
        <div class="metric-value">""" + f"{avg:.2f}" + """</div>
    </div>

    <div class="metric-card">
        <div class="metric-title">📉 Volatility</div>
        <div class="metric-value">""" + f"{volatility:.2f}" + """</div>
    </div>

</div>
""", unsafe_allow_html=True)

# ----------------------------
# AI INSIGHT
# ----------------------------
st.markdown("## 🧠 AI Scheduling Recommendation")

st.success(f"""
✔ Optimal Low-Carbon Window Identified

Start Time: **{best_window.iloc[0]['from'].strftime('%H:%M')}**

Why this window:
- Lowest carbon cluster detected
- Stable grid conditions
- Ideal for ML training execution

Expected CO₂ savings vs peak: **Significant**
""")

# ----------------------------
# GRAPH (BEST + WORST WINDOWS RESTORED)
# ----------------------------
st.markdown("## 📈 24-Hour Carbon Intelligence Curve")

fig = go.Figure()

# Main line
fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    name="Carbon Intensity",
    line=dict(width=3, color="#111827")
))

# 🟢 BEST WINDOW (LOW CARBON ZONE)
fig.add_vrect(
    x0=best_window.iloc[0]["from"],
    x1=best_window.iloc[-1]["from"],
    fillcolor="green",
    opacity=0.10,
    line_width=0,
    annotation_text="Low Carbon Window",
    annotation_position="top left"
)

# 🔴 WORST WINDOW (HIGH CARBON ZONE)
fig.add_vrect(
    x0=worst_window.iloc[0]["from"],
    x1=worst_window.iloc[-1]["from"],
    fillcolor="red",
    opacity=0.10,
    line_width=0,
    annotation_text="High Carbon Window",
    annotation_position="top left"
)

# 🔴 Peak point
fig.add_trace(go.Scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center",
    name="Peak"
))

# 🟢 Low point
fig.add_trace(go.Scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center",
    name="Low"
))

fig.update_layout(
    height=600,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Carbon Intensity (gCO₂/kWh)",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# DATA SECTION (NO BUTTONS)
# ----------------------------
st.markdown("## 📂 Forecast Intelligence Panel")

tab1, tab2, tab3 = st.tabs([
    "📊 Summary",
    "📋 Forecast Data",
    "🧠 Insights"
])

with tab1:
    st.json({
        "peak": float(peak),
        "low": float(low),
        "average": float(avg),
        "volatility": float(volatility)
    })

with tab2:
    st.dataframe(df[["from", "carbon"]], use_container_width=True)

with tab3:
    st.info("""
The UK grid shows strong temporal carbon variation.

This directly enables:
- Carbon-aware ML scheduling
- RL-based decision systems
- Dynamic workload shifting
""")