import streamlit as st

st.set_page_config(
    page_title="Carbon-Aware ML Platform",
    layout="wide"
)

# ----------------------------
# STYLING (LANDING PAGE UI)
# ----------------------------
st.markdown(
    """
    <style>
    .title {
        font-size: 50px;
        font-weight: 700;
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .subtitle {
        font-size: 18px;
        color: #aaa;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown('<div class="title">🌱 Carbon-Aware ML Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Optimize ML training using real-time grid carbon intelligence</div>',
    unsafe_allow_html=True
)

st.markdown("---")

# ----------------------------
# KPI SECTION
# ----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("🌍 Purpose", "Carbon Reduction")
col2.metric("⚡ Engine", "Scheduling AI")
col3.metric("📊 Mode", "Real-time Analytics")

st.markdown("---")

# ----------------------------
# INFO SECTION
# ----------------------------
st.info(
    "Use the sidebar to explore:\n"
    "- 📊 Overview Dashboard\n"
    "- 📈 Forecast Explorer\n"
    "- ⚙️ Simulation Lab"
)