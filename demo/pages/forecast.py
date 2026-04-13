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

# ----------------------------
# KPI SECTION
# ----------------------------
st.markdown("## 📊 Grid Intelligence Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🔥 Peak Carbon", f"{peak:.2f}")
c2.metric("🌱 Lowest Carbon", f"{low:.2f}")
c3.metric("📊 Average", f"{avg:.2f}")
c4.metric("📉 Volatility", f"{volatility:.2f}")

# ----------------------------
# INSIGHT BOX
# ----------------------------
st.markdown("## 🧠 AI Scheduling Recommendation")

st.success(f"""
✔ Optimal Low-Carbon Window Identified

Start Time: **{best_window.iloc[0]['from'].strftime('%H:%M')}**

Why this window:
- Lowest carbon cluster
- Stable grid conditions
- Best for ML workload execution

Expected CO₂ savings vs peak: **High**
""")

# ----------------------------
# GRAPH
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

# Peak marker (RED)
fig.add_trace(go.Scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center"
))

# Low marker (GREEN)
fig.add_trace(go.Scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center"
))

# Best window highlight
fig.add_vrect(
    x0=best_window.iloc[0]["from"],
    x1=best_window.iloc[2]["from"],
    fillcolor="green",
    opacity=0.08,
    line_width=0
)

fig.update_layout(
    height=600,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Carbon Intensity (gCO₂/kWh)"
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# DATA SECTION (NO BUTTON)
# ----------------------------
st.markdown("## 📂 Forecast Intelligence Panel")

tab1, tab2, tab3 = st.tabs([
    "📊 Summary",
    "📋 Forecast Data",
    "🧠 Insights"
])

with tab1:
    st.write("Key statistics from UK grid forecast.")
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
The UK grid shows strong temporal variation in carbon intensity.

This enables:
- Carbon-aware ML scheduling
- RL-based workload optimization
- Dynamic delay strategies for training jobs
""")