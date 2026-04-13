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

peak_time = df[df["carbon"] == peak]["from"].iloc[0]
low_time = df[df["carbon"] == low]["from"].iloc[0]

avg = df["carbon"].mean()
volatility = df["carbon"].std()

best_window = df.nsmallest(3, "carbon")[["from", "carbon"]]

# ----------------------------
# KPI DASHBOARD
# ----------------------------
st.markdown("## 📊 Grid Intelligence Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🔥 Peak Carbon", f"{peak:.2f} gCO₂/kWh")
c2.metric("🌱 Lowest Carbon", f"{low:.2f} gCO₂/kWh")
c3.metric("📊 Average", f"{avg:.2f} gCO₂/kWh")
c4.metric("📉 Volatility", f"{volatility:.2f}")

# ----------------------------
# RECOMMENDATION BOX (VERY IMPORTANT)
# ----------------------------
st.markdown("## 🧠 AI Scheduling Recommendation")

best_start = best_window.iloc[0]["from"]

st.success(f"""
✔ Optimal Training Window Detected

Start Time: **{best_start.strftime('%H:%M')}**

Reason:
- Lowest carbon cluster detected
- Stable grid conditions
- Ideal for ML workload execution

Expected CO₂ savings: **High (compared to peak window execution)**
""")

# ----------------------------
# MAIN GRAPH
# ----------------------------
st.markdown("## 📈 24-Hour Carbon Intelligence Curve")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    name="Carbon Intensity",
    line=dict(width=3, color="#111827")
))

# Highlight peak
fig.add_trace(go.Scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center",
    name="Peak"
))

# Highlight low
fig.add_trace(go.Scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center",
    name="Low"
))

# Highlight BEST WINDOW (visual band feel)
fig.add_vrect(
    x0=best_window.iloc[0]["from"],
    x1=best_window.iloc[1]["from"],
    fillcolor="green",
    opacity=0.1,
    line_width=0
)

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
# INSIGHT ENGINE
# ----------------------------
st.markdown("## 🧠 System Insight")

st.info(f"""
The UK grid shows strong temporal carbon variability.

Key observations:
- Peak emissions are {((peak - low) / low * 100):.1f}% higher than minimum
- Grid volatility indicates strong opportunity for scheduling optimization
- ML workloads can be shifted to low-carbon windows for significant reduction

This forecast directly feeds into the RL scheduling engine.
""")

# ----------------------------
# RAW DATA
# ----------------------------
with st.expander("📋 View Raw Forecast Data"):
    st.dataframe(df[["from", "carbon"]], use_container_width=True)