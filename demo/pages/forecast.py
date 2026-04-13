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

# -----------------
# KPI SECTION 
#  ----------------

# -----------------
# MODERN GLASS UI SECTION 
# -----------------

st.markdown("""
<style>
/* Main Container Styling */
.stApp {
    background: radial-gradient(circle at top right, #0e1726, #010409);
}

.kpi-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 20px;
    padding: 24px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    transition: all 0.3s ease-in-out;
    text-align: center;
}

.kpi-card:hover {
    transform: translateY(-5px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    box-shadow: 0 8px 40px 0 rgba(0, 0, 0, 0.5);
}

.kpi-title {
    font-size: 0.85rem;
    color: #94a3b8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 2.2rem;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    background: white;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* Dynamic Glow Accents */
.glow-red:hover { box-shadow: 0 0 20px rgba(239, 68, 68, 0.2); }
.glow-green:hover { box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
.glow-blue:hover { box-shadow: 0 0 20px rgba(59, 130, 246, 0.2); }
.glow-amber:hover { box-shadow: 0 0 20px rgba(245, 158, 11, 0.2); }

.unit {
    font-size: 0.9rem;
    color: #64748b;
    margin-left: 4px;
}
</style>
""", unsafe_allow_html=True)

def kpi_card(title, value, glow_class):
    st.markdown(f"""
    <div class="kpi-card {glow_class}">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value:.1f}<span class="unit">g</span></div>
    </div>
    """, unsafe_allow_html=True)

# -----------------
# KPI RENDER
# -----------------
col1, col2, col3, col4 = st.columns(4)
with col1: kpi_card("Peak Intensity", peak, "glow-red")
with col2: kpi_card("Daily Minimum", low, "glow-green")
with col3: kpi_card("Grid Average", avg, "glow-blue")
with col4: kpi_card("Volatility", volatility, "glow-amber")
    
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