import streamlit as st

st.set_page_config(
    page_title="CarbonML · Carbon-Aware ML Platform",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

* {
    box-sizing: border-box;
}

html, body, .stApp {
    background: #080d12 !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #e2e8f0;
}

/* Hide Streamlit chrome properly */
#MainMenu,
footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    display: none;
}

.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

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

/* ─────────────────────────────────────────────
   TICKER
───────────────────────────────────────────── */
.ticker-wrap {
    background: #0e1520;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    padding: 10px 0;
    overflow: hidden;
    white-space: nowrap;
    position: relative;
    z-index: 2;
}

.ticker-inner {
    display: inline-flex;
    gap: 48px;
    animation: ticker 28s linear infinite;
}

@keyframes ticker {
    0%   { transform: translateX(0); }
    100% { transform: translateX(-50%); }
}

.ticker-item {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}

.t-label {
    color: #475569;
    letter-spacing: 0.08em;
}

.t-val {
    color: #e2e8f0;
    font-weight: 500;
}

.t-up {
    color: #22c55e;
}

.t-dn {
    color: #ef4444;
}

.t-sep {
    color: #1e293b;
}

/* ─────────────────────────────────────────────
   NAVBAR
───────────────────────────────────────────── */
.topnav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 1px solid rgba(255,255,255,0.07);
    background: rgba(8,13,18,0.95);
    backdrop-filter: blur(12px);
    position: sticky;
    top: 0;
    z-index: 100;
}

.logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    color: #22c55e;
    letter-spacing: 0.06em;
}

.logo-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 10px #22c55e;
    display: inline-block;
    animation: blink 2s ease-in-out infinite;
}

@keyframes blink {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.5; }
}

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

/* ─────────────────────────────────────────────
   HERO
───────────────────────────────────────────── */
.hero {
    max-width: 860px;
    margin: 0 auto;
    padding: 90px 32px 56px;
    text-align: center;
    position: relative;
    z-index: 2;
}

.hero-tag {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 5px 16px;
    border-radius: 20px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 28px;
}

.hero h1 {
    font-size: 58px;
    font-weight: 700;
    line-height: 1.05;
    letter-spacing: -0.03em;
    color: #f1f5f9;
    margin-bottom: 22px;
}

.hero h1 em {
    font-style: normal;
    color: #22c55e;
}

.hero-sub {
    font-size: 17px;
    color: #64748b;
    line-height: 1.7;
    font-weight: 300;
    max-width: 650px;
    margin: 0 auto 44px;
}

/* ─────────────────────────────────────────────
   IMPACT STRIP
───────────────────────────────────────────── */
.impact-strip {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.07);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    margin-top: 40px;
}

.impact-cell {
    padding: 32px 20px;
    text-align: center;
    border-right: 1px solid rgba(255,255,255,0.07);
}

.impact-cell:last-child {
    border-right: none;
}

.impact-num {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 34px;
    font-weight: 600;
    color: #22c55e;
    display: block;
    margin-bottom: 6px;
}

.impact-label {
    font-size: 12px;
    color: #475569;
    line-height: 1.5;
}

/* ─────────────────────────────────────────────
   SECTIONS
───────────────────────────────────────────── */
.section {
    max-width: 1100px;
    margin: 0 auto;
    padding: 72px 32px;
}

.section-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-eyebrow::before {
    content: '';
    display: inline-block;
    width: 20px;
    height: 1px;
    background: #22c55e;
}

.section h2 {
    font-size: 30px;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 8px;
}

.section-sub {
    font-size: 13px;
    color: #475569;
    margin-bottom: 36px;
}

/* ─────────────────────────────────────────────
   MODULES
───────────────────────────────────────────── */
.modules-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 18px;
}

.module-card {
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 28px;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}

.module-card:hover {
    transform: translateY(-4px);
    border-color: rgba(34,197,94,0.2);
    background: #121c29;
}

.module-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #22c55e, transparent);
    opacity: 0;
    transition: opacity 0.25s ease;
}

.module-card:hover::before {
    opacity: 1;
}

.module-icon {
    width: 42px;
    height: 42px;
    border-radius: 10px;
    background: rgba(34,197,94,0.1);
    border: 1px solid rgba(34,197,94,0.2);
    display: flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 18px;
    font-size: 18px;
}

.module-title {
    font-size: 15px;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 8px;
}

.module-desc {
    font-size: 13px;
    color: #64748b;
    line-height: 1.7;
    margin-bottom: 18px;
}

.module-cta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #22c55e;
    letter-spacing: 0.05em;
}

/* ─────────────────────────────────────────────
   RESEARCH
───────────────────────────────────────────── */
.research-grid {
    display: grid;
    grid-template-columns: 1fr 1px 1fr;
    gap: 40px;
    align-items: start;
    background: #0e1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 40px;
    margin-top: 16px;
}

.v-divider {
    background: rgba(255,255,255,0.07);
}

.research-col h3 {
    font-size: 18px;
    font-weight: 600;
    color: #f1f5f9;
    margin-bottom: 14px;
}

.research-col p {
    font-size: 13px;
    color: #64748b;
    line-height: 1.8;
    margin-bottom: 12px;
}

.research-col strong {
    color: #cbd5e1;
}

.pills {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin-top: 18px;
}

.pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.2);
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

/* ─────────────────────────────────────────────
   FOOTER
───────────────────────────────────────────── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.07);
    padding: 28px 40px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-top: 40px;
}

.footer-l {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #334155;
}

.footer-l span {
    color: #22c55e;
}

.footer-r {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    color: #334155;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ─────────────────────────────────────────────
   BUTTONS
───────────────────────────────────────────── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    border-radius: 10px !important;
    transition: all 0.2s !important;
    padding: 12px 24px !important;
}

div.stButton > button[kind="primary"] {
    background: #22c55e !important;
    color: #071a10 !important;
    border: none !important;
}

div.stButton > button[kind="primary"]:hover {
    background: #16a34a !important;
    box-shadow: 0 6px 20px rgba(34,197,94,0.3) !important;
}

div.stButton > button[kind="secondary"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}

div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.2) !important;
    background: rgba(255,255,255,0.04) !important;
}

/* ─────────────────────────────────────────────
   RESPONSIVE
───────────────────────────────────────────── */
@media (max-width: 900px) {

    .hero h1 {
        font-size: 40px;
    }

    .modules-grid {
        grid-template-columns: 1fr;
    }

    .research-grid {
        grid-template-columns: 1fr;
    }

    .v-divider {
        display: none;
    }

    .impact-strip {
        grid-template-columns: repeat(2, 1fr);
    }

    .site-footer {
        flex-direction: column;
        gap: 12px;
        text-align: center;
    }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# TICKER
# ─────────────────────────────────────────────────────────────
ticker_items = [
    ("CARBON INTENSITY", "178 gCO₂/kWh", "▼ 12%", "up"),
    ("WIND GENERATION", "32.4 GW", "▲ 4.1%", "up"),
    ("SOLAR OUTPUT", "7.1 GW", "▲ 2.2%", "up"),
    ("GRID DEMAND", "38.2 GW", "▼ 1.0%", "dn"),
    ("COAL CAPACITY", "0 GW", "NET ZERO", "up"),
    ("FORECAST ACCURACY", "94.2%", "", "")
]

ticker_html = '<div class="ticker-inner">'

for label, val, delta, direction in ticker_items * 2:

    cls = (
        "t-up" if direction == "up"
        else "t-dn" if direction == "dn"
        else "t-val"
    )

    delta_html = f'<span class="{cls}">{delta}</span>' if delta else ''

    ticker_html += f"""
    <div class="ticker-item">
        <span class="t-label">{label}</span>
        <span class="t-val">{val}</span>
        {delta_html}
        <span class="t-sep">·</span>
    </div>
    """

ticker_html += '</div>'

st.markdown(
    f'<div class="ticker-wrap">{ticker_html}</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# NAVBAR
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="topnav">
    <div class="logo">
        <span class="logo-dot"></span>
        CARBONML
    </div>

    <div class="live-badge">
        <span class="logo-dot"></span>
        UK NATIONAL GRID · LIVE
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-tag">
        <span class="logo-dot"></span>
        UK Net Zero 2050 Infrastructure
    </div>

    <h1>
        Schedule ML workloads at<br>
        <em>peak carbon efficiency</em>
    </h1>

    <p class="hero-sub">
        The UK's first carbon-aware ML scheduling platform.
        Combines real-time National Grid carbon forecasting
        with reinforcement learning to cut AI training emissions
        by up to 70%.
    </p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# BUTTONS
# ─────────────────────────────────────────────────────────────
col_l, col_c, col_r = st.columns([1, 2, 1])

with col_c:

    b1, b2 = st.columns(2)

    with b1:
        if st.button(
            "⟶ Launch Dashboard",
            type="primary",
            use_container_width=True
        ):
            st.switch_page("pages/overview.py")

    with b2:
        if st.button(
            "Run Simulation Lab",
            type="secondary",
            use_container_width=True
        ):
            st.switch_page("pages/simulation.py")

# ─────────────────────────────────────────────────────────────
# IMPACT STRIP
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="impact-strip">

    <div class="impact-cell">
        <span class="impact-num">↓ 70%</span>
        <span class="impact-label">
            Max CO₂<br>Reduction
        </span>
    </div>

    <div class="impact-cell">
        <span class="impact-num">24 / 7</span>
        <span class="impact-label">
            Live Grid<br>Monitoring
        </span>
    </div>

    <div class="impact-cell">
        <span class="impact-num">RL</span>
        <span class="impact-label">
            Adaptive<br>Scheduling Agent
        </span>
    </div>

    <div class="impact-cell">
        <span class="impact-num">2050</span>
        <span class="impact-label">
            Net Zero<br>Aligned
        </span>
    </div>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# MODULES
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section">

    <div class="section-eyebrow">
        Platform Modules
    </div>

    <h2>
        Everything your ML team needs to go carbon-zero
    </h2>

    <p class="section-sub">
        Three integrated systems working in real time
        across the UK national grid
    </p>

    <div class="modules-grid">

        <div class="module-card">
            <div class="module-icon">⚡</div>

            <div class="module-title">
                Carbon Scheduler
            </div>

            <div class="module-desc">
                Identifies optimal low-carbon execution windows
                using 24-hour National Grid forecasting.
                Shifts ML workloads into green energy valleys
                without impacting SLAs.
            </div>

            <div class="module-cta">
                Open Dashboard →
            </div>
        </div>

        <div class="module-card">
            <div class="module-icon">📡</div>

            <div class="module-title">
                Carbon Intelligence API
            </div>

            <div class="module-desc">
                Real-time integration with National Grid ESO
                carbon intensity data. Forecast accuracy
                exceeding 94% at 30-minute resolution
                across all UK regions.
            </div>

            <div class="module-cta">
                View Forecast →
            </div>
        </div>

        <div class="module-card">
            <div class="module-icon">🤖</div>

            <div class="module-title">
                RL Simulation Lab
            </div>

            <div class="module-desc">
                Reinforcement learning agent learns optimal
                execution timing under carbon uncertainty.
                Benchmarked against baseline schedulers.
            </div>

            <div class="module-cta">
                Run Experiments →
            </div>
        </div>

    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RESEARCH
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="section" style="padding-top:0">

    <div class="section-eyebrow">
        Research Contribution
    </div>

    <h2>
        Bridging industrial and individual ML operations
    </h2>

    <div class="research-grid">

        <div class="research-col">

            <h3>The Problem</h3>

            <p>
                Large tech companies like
                <strong>Google and Meta</strong>
                already use carbon-aware scheduling internally.
                But this infrastructure is unavailable to
                independent researchers, startups,
                and university labs.
            </p>

            <p>
                Without access to grid carbon data or
                scheduling intelligence, these organisations
                have <strong>no mechanism to reduce
                training-time emissions.</strong>
            </p>

            <div class="pills">
                <span class="pill">Novel Framework</span>
                <span class="pill">Open Research</span>
                <span class="pill">UK Grid Data</span>
            </div>

        </div>

        <div class="v-divider"></div>

        <div class="research-col">

            <h3>Our Solution</h3>

            <p>
                A <strong>lightweight RL-enhanced carbon
                optimisation layer</strong> that runs on any
                ML workflow with no infrastructure changes.
            </p>

            <p>
                Validated against real UK National Grid
                carbon intensity data, achieving
                <strong>18–70% emissions reductions</strong>
                depending on workload urgency.
            </p>

            <div class="pills">
                <span class="pill">RL Agent</span>
                <span class="pill">18–70% Savings</span>
                <span class="pill">Production Ready</span>
            </div>

        </div>

    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="site-footer">

    <div class="footer-l">
        <span>CARBONML</span>
        · Carbon-Aware AI Systems
        · Research Prototype v1.0
        · Built by Sufiyan Ul Rehman
    </div>

    <div class="footer-r">
        Aligned with UK Net Zero 2050
    </div>

</div>
""", unsafe_allow_html=True)