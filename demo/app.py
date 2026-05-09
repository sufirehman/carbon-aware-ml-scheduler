import streamlit as st

st.set_page_config(
    page_title="CarbonML · Carbon-Aware ML Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# GLOBAL CSS (CLEANED)
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, .stApp {
    background-color: #070c12;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #e5e7eb;
}

#MainMenu, footer, header {
    visibility: hidden;
}

/* layout fix */
.block-container {
    padding: 0rem 2rem 2rem 2rem;
    max-width: 1200px;
}

/* background grid */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.03) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none;
    z-index: 0;
}

/* ================= NAVBAR ================= */
.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:16px 28px;
    background:#0b1220;
    border-bottom:1px solid rgba(255,255,255,0.06);
}

.logo {
    font-family:'IBM Plex Mono';
    color:#22c55e;
    font-weight:600;
    letter-spacing:0.08em;
    font-size:14px;
}

.status {
    font-family:'IBM Plex Mono';
    font-size:11px;
    color:#22c55e;
    padding:6px 12px;
    border-radius:999px;
    border:1px solid rgba(34,197,94,0.25);
    background:rgba(34,197,94,0.08);
}

/* ================= HERO ================= */
.hero {
    text-align:center;
    padding:80px 20px 40px;
}

.hero-tag {
    display:inline-block;
    font-family:'IBM Plex Mono';
    font-size:11px;
    color:#22c55e;
    padding:6px 14px;
    border:1px solid rgba(34,197,94,0.2);
    border-radius:999px;
    background:rgba(34,197,94,0.06);
    margin-bottom:20px;
}

.hero h1 {
    font-size:48px;
    margin-bottom:16px;
    color:#f8fafc;
}

.hero-green {
    color:#22c55e;
}

.hero-sub {
    font-size:16px;
    color:#94a3b8;
    max-width:750px;
    margin:auto;
    line-height:1.7;
}

/* ================= BUTTONS ================= */
.stButton > button {
    height:48px;
    border-radius:10px;
    font-weight:600;
    font-size:14px;
    border:1px solid rgba(255,255,255,0.08);
    transition:0.2s ease;
}

.stButton > button:hover {
    transform:translateY(-2px);
    border-color:#22c55e;
}

/* ================= IMPACT ================= */
.impact {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    margin-top:40px;
    border-top:1px solid rgba(255,255,255,0.06);
    border-bottom:1px solid rgba(255,255,255,0.06);
}

.impact-box {
    text-align:center;
    padding:28px 10px;
    border-right:1px solid rgba(255,255,255,0.06);
}

.impact-box:last-child {
    border-right:none;
}

.impact-num {
    font-family:'IBM Plex Mono';
    font-size:28px;
    color:#22c55e;
    font-weight:600;
}

.impact-label {
    font-size:12px;
    color:#94a3b8;
    margin-top:6px;
}

/* ================= CARDS ================= */
.section {
    padding:60px 20px;
}

.cards {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:18px;
    margin-top:20px;
}

.card {
    background:#0e1624;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;
    padding:22px;
    transition:0.2s;
}

.card:hover {
    transform:translateY(-3px);
    border-color:rgba(34,197,94,0.25);
}

.card-title {
    font-size:16px;
    font-weight:600;
    color:#f8fafc;
    margin-bottom:8px;
}

.card-desc {
    font-size:13px;
    color:#94a3b8;
    line-height:1.6;
}

/* ================= SIDEBAR ================= */
section[data-testid="stSidebar"] {
    background:#0b1220 !important;
    border-right:1px solid rgba(255,255,255,0.06);
}

section[data-testid="stSidebar"] * {
    color:#cbd5e1 !important;
    font-family:'IBM Plex Mono';
}

/* ================= RESPONSIVE ================= */
@media (max-width: 900px) {
    .hero h1 { font-size:34px; }
    .cards { grid-template-columns:1fr; }
    .impact { grid-template-columns:1fr 1fr; }
}

</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="navbar">
    <div class="logo">CARBONML</div>
    <div class="status">● UK GRID LIVE</div>
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR NAVIGATION (FIXED)
# =========================
with st.sidebar:
    st.title("Navigation")

    page = st.radio(
        "Go to",
        ["Home", "Dashboard", "Simulation Lab", "Forecast"]
    )

    st.markdown("---")
    st.caption("Carbon-aware ML platform using UK grid data")

# simple routing
if page == "Dashboard":
    st.switch_page("pages/overview.py")

elif page == "Simulation Lab":
    st.switch_page("pages/simulation.py")

elif page == "Forecast":
    st.switch_page("pages/forecast.py")

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">

<div class="hero-tag">UK NET ZERO 2050 PLATFORM</div>

<h1>
Schedule ML workloads at<br>
<span class="hero-green">lowest carbon intensity</span>
</h1>

<div class="hero-sub">
Carbon-aware ML scheduling system using real-time UK National Grid data
and reinforcement learning to reduce training emissions up to 70%.
</div>

</div>
""", unsafe_allow_html=True)

# =========================
# QUICK ACTIONS
# =========================
c1, c2, c3 = st.columns([1,2,1])

with c2:
    b1, b2 = st.columns(2)

    with b1:
        if st.button("Open Dashboard", use_container_width=True):
            st.switch_page("pages/overview.py")

    with b2:
        if st.button("Run Simulation", use_container_width=True):
            st.switch_page("pages/simulation.py")

# =========================
# IMPACT METRICS
# =========================
st.markdown("""
<div class="impact">

<div class="impact-box">
<div class="impact-num">70%</div>
<div class="impact-label">CO₂ Reduction</div>
</div>

<div class="impact-box">
<div class="impact-num">24/7</div>
<div class="impact-label">Live Monitoring</div>
</div>

<div class="impact-box">
<div class="impact-num">RL</div>
<div class="impact-label">Smart Scheduler</div>
</div>

<div class="impact-box">
<div class="impact-num">UK</div>
<div class="impact-label">Grid Integrated</div>
</div>

</div>
""", unsafe_allow_html=True)

# =========================
# FEATURES
# =========================
st.markdown("""
<div class="section">

<h2 style="color:#f8fafc">Platform Modules</h2>

<div class="cards">

<div class="card">
<div class="card-title">⚡ Carbon Scheduler</div>
<div class="card-desc">Finds optimal low-carbon execution windows using UK grid forecasting.</div>
</div>

<div class="card">
<div class="card-title">📡 Carbon API</div>
<div class="card-desc">Real-time National Grid ESO carbon intensity data pipeline.</div>
</div>

<div class="card">
<div class="card-title">🤖 RL Engine</div>
<div class="card-desc">Learns optimal scheduling under uncertain carbon conditions.</div>
</div>

</div>

</div>
""", unsafe_allow_html=True)