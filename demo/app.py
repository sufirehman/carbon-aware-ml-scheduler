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

/* ── TICKER ── */
.ticker-wrap {
    background: #0e1520;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 10px 0;
    overflow: hidden;
    white-space: nowrap;
}
.ticker-inner {
    display: inline-flex;
    gap: 48px;
    animation: ticker 28s linear infinite;
}
@keyframes ticker { 0%{transform:translateX(0)} 100%{transform:translateX(-50%)} }
.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}
.t-label { color: #64748b; letter-spacing: 0.08em; }
.t-val   { color: #e2e8f0; font-weight: 500; }
.t-up    { color: #4ade80; }
.t-dn    { color: #ef4444; }
.t-sep   { color: #1e293b; }

/* ── NAV ── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    background: rgba(8,13,18,0.95);
    position: sticky; top: 0; z-index: 100;
}
.logo {
    display: flex; align-items: center; gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 600;
    color: #22c55e;
    letter-spacing: 0.06em;
}
.logo-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px #22c55e;
    display: inline-block;
    animation: blink 2s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.5} }

/* ── HERO ── */
.hero {
    max-width: 860px;
    margin: 0 auto;
    padding: 80px 32px 56px;
    text-align: center;
}
.hero h1 {
    font-size: 58px; font-weight: 700;
    line-height: 1.1; letter-spacing: -0.025em;
    color: #f1f5f9; margin-bottom: 20px;
}
.hero h1 em {
    font-style: normal;
    color: #22c55e;
    text-shadow: 0 0 30px rgba(34, 197, 94, 0.2);
}
.hero-sub {
    font-size: 18px; color: #94a3b8;
    line-height: 1.7; font-weight: 300;
    max-width: 650px; margin: 0 auto 44px;
}

/* ── IMPACT STRIP (KPI IMPROVED) ── */
.impact-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    background: rgba(255, 255, 255, 0.01);
}
.impact-cell {
    padding: 40px 20px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.impact-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 38px; font-weight: 600;
    color: #22c55e; 
    display: block; margin-bottom: 6px;
    text-shadow: 0 0 15px rgba(34, 197, 94, 0.3);
}
.impact-label {
    font-size: 12px; color: #cbd5e1;
    text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── RED BUTTON STYLING ── */
[data-testid="stPageLink-NavLink"] {
    background-color: #ef4444 !important;
    color: white !important;
    border: none !important;
    padding: 12px 24px !important;
    border-radius: 8px !important;
    transition: 0.3s ease !important;
    justify-content: center !important;
}
[data-testid="stPageLink-NavLink"]:hover {
    background-color: #dc2626 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(239, 68, 68, 0.3) !important;
}

/* ── MODULES ── */
.section {
    max-width: 1060px;
    margin: 0 auto;
    padding: 64px 32px;
}
.module-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 28px;
}

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 30px 42px;
    display: flex;
    justify-content: space-between;
    background: #06090d;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TICKER
# =========================
ticker_items = [
    ("CARBON INTENSITY", "178 gCO₂/kWh", "▼ 12%", "up"),
    ("WIND GENERATION",  "32.4 GW",       "▲ 4.1%", "up"),
    ("SOLAR OUTPUT",     "7.1 GW",        "▲ 2.2%", "up"),
    ("GRID DEMAND",      "38.2 GW",       "▼ 1.0%", "dn"),
]
ticker_html = '<div class="ticker-inner">'
for label, val, delta, direction in ticker_items * 2:
    cls = "t-up" if direction == "up" else "t-dn"
    ticker_html += f'<div class="ticker-item"><span class="t-label">{label}</span> <span class="t-val">{val}</span> <span class="{cls}">{delta}</span> <span class="t-sep">·</span></div>'
ticker_html += '</div>'
st.markdown(f'<div class="ticker-wrap">{ticker_html}</div>', unsafe_allow_html=True)

# =========================
# NAVIGATION
# =========================
st.markdown("""
<div class="topnav">
    <div class="logo"><span class="logo-dot"></span> CARBONML</div>
    <div class="live-badge">UK NATIONAL GRID · LIVE</div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <h1>Schedule ML workloads at<br><em>peak carbon efficiency</em></h1>
    <p class="hero-sub">The UK's first carbon-aware ML scheduling platform. Combines real-time National Grid forecasting with reinforcement learning.</p>
</div>
""", unsafe_allow_html=True)

# RED FUNCTIONAL BUTTONS
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    b1, b2 = st.columns(2)
    with b1:
        st.page_link("pages/overview.py", label="⟶ Launch Dashboard")
    with b2:
        st.page_link("pages/simulation.py", label="Run Simulation Lab")

# =========================
# IMPACT STRIP
# =========================
st.markdown("""
<div class="impact-strip">
    <div class="impact-cell"><span class="impact-num">↓ 70%</span><span class="impact-label">Max CO₂ Reduction</span></div>
    <div class="impact-cell"><span class="impact-num">24 / 7</span><span class="impact-label">Live Grid Monitoring</span></div>
    <div class="impact-cell"><span class="impact-num">RL</span><span class="impact-label">Adaptive Agent</span></div>
    <div class="impact-cell" style="border:none;"><span class="impact-num">2050</span><span class="impact-label">Net Zero Aligned</span></div>
</div>
""", unsafe_allow_html=True)

# =========================
# MODULES SECTION
# =========================
st.markdown("""
<div class="section">
    <h2 style="color:white; margin-bottom: 30px;">Platform Modules</h2>
    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;">
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">⚡</div>
            <h3 style="color:white; font-size:18px;">Carbon Scheduler</h3>
            <p style="color:#94a3b8; font-size:14px;">Identifies optimal low-carbon execution windows.</p>
        </div>
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">📡</div>
            <h3 style="color:white; font-size:18px;">Intelligence API</h3>
            <p style="color:#94a3b8; font-size:14px;">Real-time integration with National Grid ESO data.</p>
        </div>
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">🤖</div>
            <h3 style="color:white; font-size:18px;">Simulation Lab</h3>
            <p style="color:#94a3b8; font-size:14px;">Reinforcement learning timing agent.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="site-footer">
    <div style="color:#64748b; font-size:11px;"><span>CARBONML</span> · Built by Sufiyan Ul Rehman</div>
    <div style="color:#64748b; font-size:11px;">UK NET ZERO 2050</div>
</div>
""", unsafe_allow_html=True)