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
    page_title="Forecast Explorer",
    layout="wide"
)

st.title("📈 Carbon Forecast Explorer")
st.markdown("Deep dive into UK grid carbon intensity patterns over 24 hours.")

# ----------------------------
# LOAD DATA
# ----------------------------
api = CarbonAPI()
df = api.get_24h_forecast()

# Clean data
df["carbon"] = df["actual"].fillna(df["forecast"])
df["from"] = pd.to_datetime(df["from"])

# ----------------------------
# PEAK ANALYSIS
# ----------------------------
peak = df["carbon"].max()
low = df["carbon"].min()

peak_time = df[df["carbon"] == peak]["from"].iloc[0]
low_time = df[df["carbon"] == low]["from"].iloc[0]

# ----------------------------
# KPI SECTION
# ----------------------------
st.markdown("## 📊 Key Forecast Insights")

col1, col2, col3 = st.columns(3)

col1.metric("🔥 Peak Carbon", f"{peak:.2f} gCO₂/kWh")
col2.metric("🌱 Lowest Carbon", f"{low:.2f} gCO₂/kWh")
col3.metric("📉 Variation", f"{(peak - low):.2f} gCO₂/kWh")

# ----------------------------
# MAIN INTERACTIVE GRAPH
# ----------------------------
st.markdown("## 📈 Full 24h Carbon Profile")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df["from"],
    y=df["carbon"],
    mode="lines",
    name="Carbon Intensity",
    line=dict(color="black", width=2)
))

# Highlight peak
fig.add_scatter(
    x=[peak_time],
    y=[peak],
    mode="markers+text",
    marker=dict(size=12, color="red"),
    text=["Peak"],
    textposition="top center",
    name="Peak Point"
)

# Highlight lowest
fig.add_scatter(
    x=[low_time],
    y=[low],
    mode="markers+text",
    marker=dict(size=12, color="green"),
    text=["Low"],
    textposition="bottom center",
    name="Low Point"
)

fig.update_layout(
    height=550,
    template="plotly_white",
    hovermode="x unified",
    xaxis_title="Time",
    yaxis_title="Carbon Intensity (gCO₂/kWh)",
    legend=dict(orientation="h")
)

st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# DATA TABLE VIEW
# ----------------------------
st.markdown("## 📋 Raw Forecast Data")

st.dataframe(df[["from", "carbon"]], use_container_width=True)

# ----------------------------
# INSIGHT SECTION
# ----------------------------
st.markdown("## 🧠 Insight")

st.info(
    f"""
    Carbon intensity varies significantly across the day.

    Peak demand period reaches {peak:.2f} gCO₂/kWh at {peak_time.strftime('%H:%M')}.
    Lowest carbon window occurs at {low:.2f} gCO₂/kWh at {low_time.strftime('%H:%M')}.

    This variation enables intelligent scheduling of ML workloads for carbon reduction.
    """
)