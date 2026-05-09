import streamlit as st

st.set_page_config(
    page_title="CarbonML · Carbon-Aware ML Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# GLOBAL CSS
# =========================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #080d12;
    color: #e2e8f0;
}

.stApp {
    background: #080d12;
}

#MainMenu, footer, header {
    visibility: hidden;
}

.block-container {
    padding-top: 0rem;
    padding-bottom: 0rem;
    padding-left: 0rem;
    padding-right: 0rem;
    max-width: 100%;
}

/* Background grid */
.stApp::before {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    z-index: 0;
    pointer-events: none;
}

/* NAVBAR */
.topnav {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:18px 42px;
    border-bottom:1px solid rgba(255,255,255,0.08);
    background:#0b1118;
    position:sticky;
    top:0;
    z-index:999;
}

.logo {
    color:#22c55e;
    font-family:'IBM Plex Mono', monospace;
    font-weight:600;
    letter-spacing:0.08em;
    font-size:14px;
}

.live {
    color:#22c55e;
    font-family:'IBM Plex Mono', monospace;
    font-size:11px;
    border:1px solid rgba(34,197,94,0.2);
    background:rgba(34,197,94,0.08);
    padding:6px 14px;
    border-radius:20px;
}

/* HERO */
.hero {
    text-align:center;
    padding:90px 24px 70px;
    max-width:900px;
    margin:auto;
}

.hero-tag {
    display:inline-block;
    color:#22c55e;
    font-size:11px;
    font-family:'IBM Plex Mono', monospace;
    border:1px solid rgba(34,197,94,0.2);
    background:rgba(34,197,94,0.08);
    padding:7px 16px;
    border-radius:20px;
    margin-bottom:30px;
    letter-spacing:0.08em;
}

.hero h1 {
    font-size:58px;
    line-height:1.05;
    color:#f8fafc;
    margin-bottom:22px;
}

.hero-green {
    color:#22c55e;
}

.hero-sub {
    color:#64748b;
    font-size:18px;
    line-height:1.8;
    max-width:700px;
    margin:auto;
}

/* BUTTONS */
.stButton > button {
    height:52px;
    border-radius:10px;
    font-weight:600;
    font-size:14px;
    transition:0.2s;
}

.stButton > button:hover {
    transform:translateY(-2px);
}

/* IMPACT */
.impact-row {
    display:grid;
    grid-template-columns:repeat(4,1fr);
    border-top:1px solid rgba(255,255,255,0.07);
    border-bottom:1px solid rgba(255,255,255,0.07);
}

.impact-box {
    text-align:center;
    padding:34px 20px;
    border-right:1px solid rgba(255,255,255,0.07);
}

.impact-box:last-child {
    border-right:none;
}

.impact-num {
    color:#22c55e;
    font-size:34px;
    font-family:'IBM Plex Mono', monospace;
    font-weight:600;
}

.impact-label {
    color:#64748b;
    font-size:13px;
    margin-top:8px;
}

/* SECTION */
.section {
    max-width:1100px;
    margin:auto;
    padding:80px 30px;
}

.section-eyebrow {
    color:#22c55e;
    font-size:11px;
    font-family:'IBM Plex Mono', monospace;
    margin-bottom:10px;
    text-transform:uppercase;
    letter-spacing:0.12em;
}

.section h2 {
    font-size:34px;
    color:#f8fafc;
    margin-bottom:12px;
}

.section-sub {
    color:#64748b;
    margin-bottom:40px;
}

/* CARDS */
.card-grid {
    display:grid;
    grid-template-columns:repeat(3,1fr);
    gap:20px;
}

.card {
    background:#0f1722;
    border:1px solid rgba(255,255,255,0.07);
    border-radius:14px;
    padding:30px;
    transition:0.25s;
}

.card:hover {
    transform:translateY(-4px);
    border-color:rgba(34,197,94,0.2);
}

.card-icon {
    font-size:28px;
    margin-bottom:18px;
}

.card-title {
    font-size:18px;
    color:#f8fafc;
    margin-bottom:12px;
    font-weight:600;
}

.card-desc {
    color:#64748b;
    line-height:1.7;
    font-size:14px;
}

/* FOOTER */
.footer {
    border-top:1px solid rgba(255,255,255,0.07);
    margin-top:60px;
    padding:26px 40px;
    display:flex;
    justify-content:space-between;
    color:#94a3b8;
    font-size:12px;
}

@media(max-width:900px){

    .hero h1 {
        font-size:42px;
    }

    .card-grid {
        grid-template-columns:1fr;
    }

    .impact-row {
        grid-template-columns:1fr 1fr;
    }

    .footer {
        flex-direction:column;
        gap:10px;
    }
}

</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="topnav">
    <div class="logo">CARBONML</div>
    <div class="live">● UK NATIONAL GRID · LIVE</div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">

<div class="hero-tag">
UK NET ZERO 2050 INFRASTRUCTURE
</div>

<h1>
Schedule ML workloads at<br>
<span class="hero-green">peak carbon efficiency</span>
</h1>

<div class="hero-sub">
The UK's first carbon-aware ML scheduling platform.
Combines real-time National Grid forecasting with reinforcement learning
to reduce AI training emissions by up to 70%.
</div>

</div>
""", unsafe_allow_html=True)

# =========================
# BUTTONS
# =========================
c1, c2, c3 = st.columns([1,2,1])

with c2:
    b1, b2 = st.columns(2)

    with b1:
        if st.button("Launch Dashboard", use_container_width=True):
            st.switch_page("pages/overview.py")

    with b2:
        if st.button("Simulation Lab", use_container_width=True):
            st.switch_page("pages/simulation.py")

# =========================
# IMPACT STRIP
# =========================
st.markdown("""
<div class="impact-row">

<div class="impact-box">
<div class="impact-num">↓ 70%</div>
<div class="impact-label">Maximum CO₂ Reduction</div>
</div>

<div class="impact-box">
<div class="impact-num">24/7</div>
<div class="impact-label">Live Grid Monitoring</div>
</div>

<div class="impact-box">
<div class="impact-num">RL</div>
<div class="impact-label">Adaptive Scheduling Agent</div>
</div>

<div class="impact-box">
<div class="impact-num">2050</div>
<div class="impact-label">Net Zero Aligned</div>
</div>

</div>
""", unsafe_allow_html=True)

# =========================
# MODULES
# =========================
st.markdown("""
<div class="section">

<div class="section-eyebrow">
Platform Modules
</div>

<h2>Everything your ML team needs</h2>

<div class="section-sub">
Three integrated systems working together in real time
across the UK national grid.
</div>

<div class="card-grid">

<div class="card">
<div class="card-icon">⚡</div>
<div class="card-title">Carbon Scheduler</div>
<div class="card-desc">
Identifies optimal low-carbon execution windows using
24-hour National Grid forecasting.
</div>
</div>

<div class="card">
<div class="card-icon">📡</div>
<div class="card-title">Carbon Intelligence API</div>
<div class="card-desc">
Real-time integration with National Grid ESO carbon data
with high forecast accuracy.
</div>
</div>

<div class="card">
<div class="card-icon">🤖</div>
<div class="card-title">RL Simulation Lab</div>
<div class="card-desc">
Reinforcement learning agent learns optimal execution timing
under carbon uncertainty.
</div>
</div>

</div>
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
<div>
CARBONML · Carbon-Aware AI Systems · Built by Sufiyan Ul Rehman
</div>

<div>
Aligned with UK Net Zero 2050
</div>
</div>
""", unsafe_allow_html=True)