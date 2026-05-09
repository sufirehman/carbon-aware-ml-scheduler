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
.live-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.25);
    padding: 5px 14px;
    border-radius: 20px;
    display: inline-flex;
    align-items: center;
    gap: 7px;
}

/* ── HERO ── */
.hero {
    max-width: 860px;
    margin: 0 auto;
    padding: 80px 32px 56px;
    text-align: center;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 5px 16px; border-radius: 20px;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 28px;
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

/* ── IMPACT STRIP ── */
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
    transition: 0.3s ease;
}
.impact-cell:hover { background: rgba(34, 197, 94, 0.04); }
.impact-cell:last-child { border-right: none; }

.impact-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 38px; font-weight: 600;
    color: #4ade80;
    display: block; margin-bottom: 6px;
    letter-spacing: -0.02em;
    text-shadow: 0 0 20px rgba(34, 197, 94, 0.3);
}
.impact-label {
    font-size: 12px; color: #cbd5e1;
    line-height: 1.5; text-transform: uppercase; letter-spacing: 0.05em;
}

/* ── MODULES ── */
.section {
    max-width: 1060px;
    margin: 0 auto;
    padding: 64px 32px;
}
.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #22c55e;
    letter-spacing: 0.12em; text-transform: uppercase;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 10px;
}
.section-eyebrow::before {
    content: ''; display: inline-block;
    width: 20px; height: 1px; background: #22c55e;
}
.section h2 {
    font-size: 32px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.01em;
    margin-bottom: 6px;
}
.section-sub { font-size: 14px; color: #94a3b8; margin-bottom: 36px; }

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
    position: relative;
    overflow: hidden;
    transition: all 0.25s;
}
.module-card:hover {
    border-color: rgba(34,197,94,0.3);
    background: #131c29;
    transform: translateY(-2px);
}
.module-title {
    font-size: 16px; font-weight: 600;
    color: #f1f5f9; margin-bottom: 8px;
}
.module-desc {
    font-size: 14px; color: #94a3b8;
    line-height: 1.65; margin-bottom: 18px;
}

/* ── RESEARCH BLOCK ── */
.research-grid {
    display: grid;
    grid-template-columns: 1fr 1px 1fr;
    gap: 40px;
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 40px;
    margin-top: 16px;
}
.v-divider { background: rgba(255,255,255,0.07); }

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 30px 42px;
    display: flex;
    justify-content: space-between;
    background: #06090d;
    margin-top: 40px;
}
.footer-l { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #64748b; }
.footer-l span { color: #22c55e; }

/* Streamlit button overrides */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
}
</style>
""", unsafe_allow_html=True)

# =========================
# TICKER COMPONENT
# =========================
ticker_items = [
    ("CARBON INTENSITY", "178 gCO₂/kWh", "▼ 12%", "up"),
    ("WIND GENERATION",  "32.4 GW",       "▲ 4.1%", "up"),
    ("SOLAR OUTPUT",     "7.1 GW",        "▲ 2.2%", "up"),
    ("GRID DEMAND",      "38.2 GW",       "▼ 1.0%", "dn"),
    ("COAL CAPACITY",    "0 GW",          "NET ZERO", "up"),
]
ticker_html = '<div class="ticker-inner">'
for label, val, delta, direction in ticker_items * 2:
    cls = "t-up" if direction == "up" else ("t-dn" if direction == "dn" else "t-val")
    ticker_html += f'''
        <div class="ticker-item">
            <span class="t-label">{label}</span>
            <span class="t-val">{val}</span>
            <span class="{cls}">{delta}</span>
            <span class="t-sep">·</span>
        </div>'''
ticker_html += '</div>'
st.markdown(f'<div class="ticker-wrap">{ticker_html}</div>', unsafe_allow_html=True)

# =========================
# NAVIGATION
# =========================
st.markdown("""
<div class="topnav">
    <div class="logo"><span class="logo-dot"></span> CARBONML</div>
    <div class="live-badge"><span class="logo-dot"></span> UK NATIONAL GRID · LIVE</div>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO SECTION
# =========================
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="logo-dot"></span> UK Net Zero 2050 Infrastructure</div>
    <h1>Schedule ML workloads at<br><em>peak carbon efficiency</em></h1>
    <p class="hero-sub">
        The UK's first carbon-aware ML scheduling platform. Combines real-time National Grid
        carbon forecasting with reinforcement learning to cut AI training emissions by up to 70%.
    </p>
</div>
""", unsafe_allow_html=True)

# =========================
# HERO BUTTONS
# =========================
col_l, col_c, col_r = st.columns([1, 2, 1])
with col_c:
    b1, b2 = st.columns(2)
    
    with b1:
        if st.button("⟶ Launch Dashboard", 
                    type="primary", 
                    use_container_width=True,
                    key="launch_btn"):
            st.switch_page("pages/overview.py")
    
    with b2:
        if st.button("Run Simulation Lab", 
                    type="secondary", 
                    use_container_width=True,
                    key="sim_btn"):
            st.switch_page("pages/simulation.py")

# =========================
# IMPACT STRIP
# =========================
st.markdown("""
<div class="impact-strip">
    <div class="impact-cell">
        <span class="impact-num">↓ 70%</span>
        <span class="impact-label">Max CO₂<br>Reduction</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">24 / 7</span>
        <span class="impact-label">Live Grid<br>Monitoring</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">RL</span>
        <span class="impact-label">Adaptive<br>Scheduling Agent</span>
    </div>
    <div class="impact-cell">
        <span class="impact-num">2050</span>
        <span class="impact-label">Net Zero<br>Aligned</span>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# MODULES SECTION
# =========================
st.markdown("""
<div class="section">
    <div class="section-eyebrow">Platform Modules</div>
    <h2>Everything your ML team needs to go carbon-zero</h2>
    <p class="section-sub">Three integrated systems working in real time across the UK national grid</p>
    <div class="modules-grid">
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">⚡</div>
            <div class="module-title">Carbon Scheduler</div>
            <div class="module-desc">Identifies optimal low-carbon execution windows using 24-hour National Grid forecasting.</div>
        </div>
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">📡</div>
            <div class="module-title">Carbon Intelligence API</div>
            <div class="module-desc">Real-time integration with National Grid ESO carbon data with 94%+ forecast accuracy.</div>
        </div>
        <div class="module-card">
            <div style="font-size:24px; margin-bottom:15px;">🤖</div>
            <div class="module-title">RL Simulation Lab</div>
            <div class="module-desc">Reinforcement learning agent learns optimal execution timing under carbon uncertainty.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# RESEARCH CONTRIBUTION
# =========================
st.markdown("""
<div class="section" style="padding-top:0">
    <div class="section-eyebrow">Research Contribution</div>
    <h2>Bridging industrial and individual ML operations</h2>
    <div class="research-grid">
        <div class="research-col">
            <h3 style="color:#f1f5f9; font-size:18px;">The Problem</h3>
            <p style="color:#94a3b8; font-size:14px; line-height:1.6;">Google and Meta use carbon-aware scheduling internally, but this isn't available to independent researchers or university labs.</p>
        </div>
        <div class="v-divider"></div>
        <div class="research-col">
            <h3 style="color:#f1f5f9; font-size:18px;">Our Solution</h3>
            <p style="color:#94a3b8; font-size:14px; line-height:1.6;">A lightweight RL-enhanced carbon optimisation layer that shifts execution to green grid windows automatically.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="site-footer">
    <div class="footer-l"><span>CARBONML</span> · Carbon-Aware AI Systems · Built by Sufiyan Ul Rehman</div>
    <div class="footer-l">Aligned with UK Net Zero 2050</div>
</div>
""", unsafe_allow_html=True)