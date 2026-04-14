import streamlit as st

st.set_page_config(
    page_title="Carbon-Aware ML Platform",
    layout="wide"
)

# ----------------------------
# GLOBAL STYLE (UPGRADED)
# ----------------------------
st.markdown(
    """
    <style>

    .stApp {
        background: radial-gradient(circle at top, #0b1220, #020617);
        color: white;
    }

    .hero-title {
        font-size: 64px;
        font-weight: 900;
        text-align: center;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 40px;
    }

    .hero-subtitle {
        font-size: 20px;
        text-align: center;
        color: #cbd5e1;
        margin-bottom: 10px;
    }

    .metric-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 20px;
        border-radius: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 22px;
        border-radius: 18px;
        text-align: center;
        transition: 0.3s;
    }

    .card:hover {
        transform: translateY(-6px);
        border-color: rgba(0, 201, 255, 0.5);
    }

    div.stButton > button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: black !important;
        font-weight: 700;
        border-radius: 10px;
        width: 100%;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HERO
# ----------------------------
st.markdown('<div class="hero-title">🌱 Carbon-Aware ML Intelligence</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero-subtitle">Reducing AI training emissions using real-time UK grid carbon forecasting + RL scheduling</div>',
    unsafe_allow_html=True
)

# ----------------------------
# IMPACT METRICS (VERY IMPORTANT)
# ----------------------------
st.markdown("## 📊 System Impact Overview")

c1, c2, c3, c4 = st.columns(4)

c1.markdown('<div class="metric-card"><h2>↓ 18–70%</h2><p>CO₂ Reduction Potential</p></div>', unsafe_allow_html=True)
c2.markdown('<div class="metric-card"><h2>⚡ Real-Time</h2><p>Grid Carbon Awareness</p></div>', unsafe_allow_html=True)
c3.markdown('<div class="metric-card"><h2>🤖 RL-Based</h2><p>Adaptive Scheduling</p></div>', unsafe_allow_html=True)
c4.markdown('<div class="metric-card"><h2>🌍 UK Grid</h2><p>Live Carbon Intensity Data</p></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ----------------------------
# CTA
# ----------------------------
col1, col2, col3 = st.columns([1,2,1])

with col2:
    b1, b2 = st.columns(2)

    with b1:
        if st.button("🚀 Launch Simulation Engine"):
            st.switch_page("pages/overview.py")

    with b2:
        if st.button("📊 View Carbon Dashboard"):
            st.switch_page("pages/overview.py")

# ----------------------------
# FEATURES
# ----------------------------
st.markdown("## ⚙️ Core System Modules")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="card">
    ⚡<br><br>
    <b>Carbon Scheduler</b><br>
    Finds optimal low-carbon execution windows using grid forecasting
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="card">
    🤖<br><br>
    <b>Reinforcement Learning Agent</b><br>
    Learns optimal execution timing under carbon uncertainty
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="card">
    📡<br><br>
    <b>Carbon Intelligence API</b><br>
    Integrates real-time UK grid emissions forecasting
    </div>
    """, unsafe_allow_html=True)

# ----------------------------
# WHY THIS MATTERS
# ----------------------------
st.markdown("## 🌍 Research Contribution")

st.markdown("""
<div style="text-align:center; font-size:18px; color:#cbd5e1; max-width:900px; margin:auto;">

This system demonstrates a **novel carbon-aware ML scheduling framework** that bridges the gap between:

- High-scale industrial schedulers (Google/Meta systems)
- Individual ML practitioners and research labs

<br>

It introduces a **lightweight RL-enhanced carbon optimization layer** for real-world ML training workflows using UK grid carbon intensity data.

</div>
""", unsafe_allow_html=True)

# ----------------------------
# FOOTER
# ----------------------------
st.markdown("<br><br>", unsafe_allow_html=True)

st.markdown("""
<div style="text-align:center; color:#64748b;">
Built by <b>Sufiyan Ul Rehman</b> • Carbon-Aware AI Systems • Research Prototype v1.0
</div>
""", unsafe_allow_html=True)