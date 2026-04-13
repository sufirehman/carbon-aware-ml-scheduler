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
# HEADER (UPGRADED)
# ----------------------------
st.markdown("""
<div style="text-align:center; padding-bottom:10px;">
<h1 style="font-size:52px; background:linear-gradient(90deg,#00C9FF,#92FE9D);
-webkit-background-clip:text; -webkit-text-fill-color:transparent;">
📡 Carbon Intelligence Forecast System
</h1>
<p style="color:#94a3b8; font-size:18px;">
AI-powered grid carbon forecasting for ML workload scheduling optimization
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

# best and worst windows (more stable than nsmallest)
best_window = df.nsmallest(3, "carbon").sort_values("from")
worst_window = df.nlargest(3, "carbon").sort_values("from")

best_start, best_end = best_window["from"].iloc[0], best_window["from"].iloc[-1]
worst_start, worst_end = worst_window["from"].iloc[0], worst_window["from"].iloc[-1]

# ----------------------------
# KPI DASHBOARD (UPGRADED)
# ----------------------------
st.markdown("## 📊 Grid Intelligence Overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🔥 Peak Carbon", f"{peak:.2f} gCO₂/kWh")
c2.metric("🌱 Lowest Carbon", f"{low:.2f} gCO₂/kWh")
c3.metric("📊 Avg Carbon", f"{avg:.2f} gCO₂/kWh")
c4.metric("📉 Volatility", f"{volatility:.2f}")

# ----------------------------
# DECISION PANEL (KEY UPGRADE)
# ----------------------------
st.markdown("## 🧠 Scheduling Intelligence Output")

st.success(f"""
✔ **Recommended Action: Schedule ML Training in Low-Carbon Window**

🟢 Best Start: {best_start.strftime('%H:%M')}  
🔴 Avoid High Carbon Window: {worst_start.strftime('%H:%M')}

Expected outcome:
- Reduced CO₂ emissions
- Improved carbon efficiency
- Optimal use of grid conditions
""")

# ----------------------------
# MAIN GRAPH (CONTROL SYSTEM STYLE)
# ----------------------------
st.markdown("## 📈 Carbon Intensity Control View (24h)")

fig = go.Figure()

# main signal
fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    name="Grid Carbon Intensity",
    line=dict(width=3, color="#0f172a")
))

# peak
fig.add_trace(go.Scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center",
    name="Peak"
))

# low
fig.add_trace(go.Scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center",
    name="Low"
))

# BEST WINDOW (green band)
fig.add_vrect(
    x0=best_start,
    x1=best_end,
    fillcolor="green",
    opacity=0.15,
    line_width=0,
    annotation_text="Optimal Execution Window",
    annotation_position="top left"
)

# WORST WINDOW (red band)
fig.add_vrect(
    x0=worst_start,
    x1=worst_end,
    fillcolor="red",
    opacity=0.10,
    line_width=0,
    annotation_text="High Carbon Window",
    annotation_position="bottom left"
)

fig.update_layout(
    height=620,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Carbon Intensity (gCO₂/kWh)",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# FINAL INSIGHT (NO EXPANDER NEEDED)
# ----------------------------
st.markdown("## 🧠 System Insight")

st.info(f"""
The UK grid exhibits strong temporal carbon variability.

Key findings:
- Peak emissions are {((peak - low) / low * 100):.1f}% higher than minimum
- Significant scheduling opportunity exists within low-carbon clusters
- This directly enables RL-based carbon-aware ML training optimization

This forecast module acts as the **input intelligence layer** for the scheduling engine.
""")