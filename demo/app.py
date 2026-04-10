import streamlit as st

st.set_page_config(
    page_title="Carbon-Aware ML Platform",
    layout="wide"
)

# ----------------------------
# STARTUP-STYLE GLOBAL CSS
# ----------------------------
st.markdown(
    """
    <style>

    .stApp {
        background: radial-gradient(circle at top, #0f172a, #020617);
        color: white;
    }

    .hero-title {
        font-size: 60px;
        font-weight: 800;
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
        margin-top: 10px;
        margin-bottom: 30px;
    }

    .card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 25px;
        border-radius: 18px;
        backdrop-filter: blur(12px);
        text-align: center;
        transition: 0.3s;
    }

    .card:hover {
        transform: translateY(-5px);
        border-color: rgba(0, 201, 255, 0.5);
    }

    /* 🔥 FIX BUTTON STYLING */
    div.stButton > button {
        background: linear-gradient(90deg, #00C9FF, #92FE9D);
        color: black !important;
        font-weight: 700;
        padding: 0.6rem 1.2rem;
        border-radius: 10px;
        border: none;
        width: 100%;
        transition: 0.3s ease;
    }

    div.stButton > button:hover {
        transform: scale(1.03);
        box-shadow: 0 0 15px rgba(0, 201, 255, 0.5);
        cursor: pointer;
    }

    div.stButton > button:focus {
        outline: none;
    }

    </style>
    """,
    unsafe_allow_html=True
)
# ----------------------------
# HERO SECTION
# ----------------------------
st.markdown('<div class="hero-title">🌱 Carbon-Aware ML Platform</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="hero-subtitle">Optimize machine learning training using real-time UK grid carbon intelligence</div>',
    unsafe_allow_html=True
)

st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------------------
# CTA BUTTONS (SAME ROW)
# ----------------------------
col1, col2, col3 = st.columns([1,1,1])

with col2:
    b1, b2 = st.columns(2)

    with b1:
        if st.button("🚀 Launch Platform"):
            st.switch_page("pages/overview.py")

    with b2:
        if st.button("📊 View Live Dashboard"):
            st.switch_page("pages/overview.py")

# ----------------------------
# FEATURE CARDS (STARTUP STYLE)
# ----------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="card">
        ⚡<br><br>
        <b>Smart Scheduling Engine</b><br>
        Finds optimal low-carbon training windows
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="card">
        🌍<br><br>
        <b>Real-Time Carbon Data</b><br>
        Uses UK grid intensity forecasting API
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        """
        <div class="card">
        📊<br><br>
        <b>Impact Analytics</b><br>
        Measures CO₂ savings vs immediate training
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br><br>", unsafe_allow_html=True)

# ----------------------------
# PRODUCT VALUE SECTION
# ----------------------------
st.markdown(
    """
    <div style="text-align:center; font-size:22px; color:#e2e8f0; max-width:900px; margin:auto;">

    Built by <b>Sufiyan</b> as a carbon-aware AI infrastructure system for 
    <b>sustainable machine learning training optimization</b>.<br>

    This platform demonstrates how <b>real-time grid carbon intelligence</b> can be used to 
    intelligently schedule ML workloads and reduce unnecessary CO₂ emissions in AI systems.

    </div>
    """,
    unsafe_allow_html=True
)
