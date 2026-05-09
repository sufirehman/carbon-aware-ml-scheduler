import streamlit as st

st.set_page_config(
    page_title="CarbonML · Carbon-Aware ML Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# GLOBAL CSS & ENHANCEMENTS
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

* { box-sizing: border-box; }

html, body, .stApp {
    background: #080d12 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #e2e8f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Grid background */
.stApp::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 48px 48px;
    pointer-events: none;
    z-index: 0;
}

/* =========================
   NAVBAR
========================= */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    background: rgba(8,13,18,0.95);
    position: sticky;
    top: 0;
    z-index: 100;
}

.logo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #22c55e;
}

/* =========================
   HERO
========================= */
.hero {
    max-width: 860px;
    margin: 0 auto;
    padding: 80px 32px 40px;
    text-align: center;
}

.hero-tag {
    display: inline-block;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 6px 16px;
    border-radius: 20px;
    margin-bottom: 24px;
}

.hero h1 {
    font-size: 58px;
    font-weight: 700;
    line-height: 1.1;
    color: #f1f5f9;
    margin-bottom: 18px;
}

.hero em {
    font-style: normal;
    color: #22c55e;
}

.hero-sub {
    font-size: 18px;
    color: #94a3b8;
    max-width: 650px;
    margin: 0 auto 40px;
}

/* =========================
   BUTTONS (FIXED + CENTERED)
========================= */
.hero-btn-wrap {
    display: flex;
    justify-content: center;
    gap: 16px;
    margin-top: 10px;
}

/* Streamlit page link styling */
div[data-testid="stPageLink"] a {
    width: 100%;
    display: flex;
    justify-content: center;
    align-items: center;
    text-decoration: none !important;

    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;

    padding: 14px 22px !important;
    border-radius: 10px !important;

    transition: all 0.2s ease !important;
}

/* PRIMARY */
.primary-link a {
    background: #22c55e !important;
    color: #071a10 !important;
    border: none !important;
    box-shadow: 0 6px 18px rgba(34,197,94,0.18);
}

.primary-link a:hover {
    background: #16a34a !important;
    transform: translateY(-2px);
}

/* SECONDARY */
.secondary-link a {
    background: rgba(255,255,255,0.03) !important;
    color: #cbd5e1 !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
}

.secondary-link a:hover {
    background: rgba(255,255,255,0.06) !important;
    color: #ffffff !important;
    transform: translateY(-2px);
}

/* =========================
   IMPACT STRIP
========================= */
.impact-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
}

.impact-cell {
    padding: 40px 20px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.impact-cell:last-child { border-right: none; }

.impact-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 38px;
    color: #4ade80;
    display: block;
}

.impact-label {
    font-size: 12px;
    color: #cbd5e1;
    text-transform: uppercase;
}

/* =========================
   MODULES
========================= */
.section {
    max-width: 1060px;
    margin: 0 auto;
    padding: 64px 32px;
}

.section h2 {
    font-size: 32px;
    color: #f1f5f9;
}

.modules-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
}

.module-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 28px;
}

/* =========================
   FOOTER
========================= */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 30px 42px;
    display: flex;
    justify-content: space-between;
    background: #06090d;
    margin-top: 40px;
    font-size: 11px;
    color: #64748b;
}

.site-footer span { color: #22c55e; }

</style>
""", unsafe_allow_html=True)

# =========================
# NAVBAR
# =========================
st.markdown("""
<div class="topnav">
    <div class="logo">CARBONML</div>
    <div class="logo">UK GRID · LIVE</div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-tag">UK NET ZERO 2050 INFRASTRUCTURE</div>

    <h1>
        Schedule ML workloads at<br>
        <em>peak carbon efficiency</em>
    </h1>

    <div class="hero-sub">
        The UK's first carbon-aware ML scheduling platform.
        Combines real-time National Grid forecasting with reinforcement learning
        to reduce AI training emissions by up to 70%.
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# ✅ FIXED BUTTON SECTION (CENTERED + WORKING)
# =========================
col_l, col_c, col_r = st.columns([1, 2, 1])

with col_c:
    st.markdown('<div class="hero-btn-wrap">', unsafe_allow_html=True)

    b1, b2 = st.columns(2)

    with b1:
        st.markdown('<div class="primary-link">', unsafe_allow_html=True)
        st.page_link("pages/overview.py", label="⟶ Launch Dashboard")
        st.markdown('</div>', unsafe_allow_html=True)

    with b2:
        st.markdown('<div class="secondary-link">', unsafe_allow_html=True)
        st.page_link("pages/simulation.py", label="🧪 Run Simulation Lab")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# IMPACT STRIP
# =========================
st.markdown("""
<div class="impact-strip">
    <div class="impact-cell">
        <span class="impact-num">↓ 70%</span>
        <span class="impact-label">Max CO₂ Reduction</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">24 / 7</span>
        <span class="impact-label">Live Grid Monitoring</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">RL</span>
        <span class="impact-label">Adaptive Scheduler</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">2050</span>
        <span class="impact-label">Net Zero Aligned</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MODULES
# =========================
st.markdown("""
<div class="section">
    <h2>Platform Modules</h2>

    <div class="modules-grid">
        <div class="module-card">
            ⚡ Carbon Scheduler
        </div>

        <div class="module-card">
            📡 Carbon Intelligence API
        </div>

        <div class="module-card">
            🤖 RL Simulation Lab
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="site-footer">
    <div><span>CARBONML</span> · Built by Sufiyan Ul Rehman</div>
    <div>UK Net Zero 2050</div>
</div>
""", unsafe_allow_html=True)