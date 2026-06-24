import streamlit as st

st.set_page_config(
    page_title="CAML-TC · Carbon-Aware ML Scheduler",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #05090f !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #cbd5e1;
    overflow-x: hidden;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 52px 52px;
}

/* ── NAVIGATION ── */
.top-nav {
    position: sticky; top: 0; z-index: 50;
    display: flex; align-items: center; justify-content: space-between;
    padding: 14px 48px;
    background: rgba(5,9,15,0.92);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    backdrop-filter: blur(12px);
}
.nav-brand {
    display: flex; align-items: center; gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 14px; font-weight: 600; color: #f1f5f9;
    letter-spacing: 0.06em;
}
.brand-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: #22c55e; flex-shrink: 0;
}
.nav-links {
    display: flex; gap: 32px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; letter-spacing: 0.05em;
}
.nav-link {
    color: #94a3b8 !important;
    text-decoration: none !important;
    transition: color 0.2s;
}
.nav-link:hover { color: #22c55e !important; }
.nav-link:focus-visible { color: #22c55e !important; outline: none; }
.nav-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #475569;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    padding: 5px 14px; border-radius: 6px;
    letter-spacing: 0.05em;
}

/* ── HERO ── */
.hero-wrap {
    max-width: 860px; margin: 0 auto;
    padding: 80px 32px 60px;
    position: relative; z-index: 1;
}
.hero-cite-top {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #334155;
    letter-spacing: 0.06em; margin-bottom: 28px;
}
.hero-cite-top a { color: #475569; text-decoration: none; }
.hero-cite-top a:hover { color: #22c55e; }
.hero-h1 {
    font-size: 40px; font-weight: 700;
    line-height: 1.15; letter-spacing: -0.025em;
    color: #f1f5f9; margin-bottom: 20px;
}
.hero-sub {
    font-size: 16px; font-weight: 300;
    color: #64748b; line-height: 1.8;
    max-width: 640px; margin-bottom: 12px;
}
.hero-install {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #334155;
    letter-spacing: 0.04em; margin-bottom: 36px;
}
.hero-install code { color: #475569; }
.hero-cta {
    display: flex; align-items: center; gap: 14px;
}
.btn-primary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 500; letter-spacing: 0.04em;
    color: #051a0d; background: #22c55e;
    border: none; border-radius: 6px;
    padding: 10px 24px; text-decoration: none;
    transition: background 0.2s; display: inline-block;
}
.btn-primary:hover { background: #16a34a; }
.btn-primary:focus-visible { outline: 2px solid rgba(34,197,94,0.5); outline-offset: 2px; }
.btn-secondary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; font-weight: 400; letter-spacing: 0.04em;
    color: #64748b; background: transparent;
    border: 1px solid rgba(255,255,255,0.1); border-radius: 6px;
    padding: 10px 24px; text-decoration: none;
    transition: color 0.2s, border-color 0.2s; display: inline-block;
}
.btn-secondary:hover { color: #e2e8f0 !important; border-color: rgba(255,255,255,0.2); }
.btn-secondary:focus-visible { outline: 2px solid rgba(255,255,255,0.2); outline-offset: 2px; }

/* ── IMPACT STRIP ── */
.impact-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.012);
    position: relative; z-index: 1;
}
.impact-cell {
    padding: 40px 20px; text-align: center;
    border-right: 1px solid rgba(255,255,255,0.06);
    transition: background 0.2s;
}
.impact-cell:last-child { border-right: none; }
.impact-cell:hover { background: rgba(34,197,94,0.025); }
.impact-n {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 36px; font-weight: 600;
    color: #22c55e; display: block; margin-bottom: 8px;
    letter-spacing: -0.02em;
}
.impact-l {
    font-size: 11px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.07em;
    line-height: 1.55;
}

/* ── SECTIONS ── */
.section { max-width: 1080px; margin: 0 auto; padding: 64px 32px; position: relative; z-index: 1; }
.section-eye {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 10px;
}
.section-eye::before { content:''; display:inline-block; width:24px; height:1px; background:#22c55e; }
.section-h2 {
    font-size: 28px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.015em; margin-bottom: 6px;
}
.section-sub { font-size: 13px; color: #64748b; margin-bottom: 36px; }

/* ── PIPELINE ── */
.pipeline {
    display: grid; grid-template-columns: 1fr 28px 1fr 28px 1fr 28px 1fr;
    align-items: center; margin: 40px 0;
}
.pipe-step {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 24px 18px;
    text-align: center; transition: border-color 0.2s;
}
.pipe-step:hover { border-color: rgba(34,197,94,0.2); }
.pipe-icon { margin-bottom: 14px; display: flex; justify-content: center; }
.pipe-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #94a3b8;
    font-weight: 500; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.pipe-desc { font-size: 12px; color: #475569; line-height: 1.6; }
.pipe-arrow { text-align: center; color: #1e3a2a; font-size: 16px; }

/* ── FEATURE GRID ── */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.feat-card {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 26px 22px;
    transition: border-color 0.2s;
}
.feat-card:hover { border-color: rgba(255,255,255,0.14); }
.feat-icon { margin-bottom: 14px; }
.feat-title { font-size: 14px; font-weight: 600; color: #e2e8f0; margin-bottom: 8px; }
.feat-body { font-size: 13px; color: #64748b; line-height: 1.7; }

/* ── RESULTS TABLE ── */
.results-strip {
    background: #040810;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 56px 0;
}
.results-inner { max-width: 1080px; margin: 0 auto; padding: 0 32px; }
.results-table {
    width: 100%; border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 28px;
}
.results-table th {
    font-size: 10px; color: #334155;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0 20px 12px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.results-table td {
    font-size: 13px; color: #94a3b8;
    padding: 14px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.results-table tr:last-child td { border-bottom: none; }
.td-strategy { color: #f1f5f9; font-weight: 500; }
.td-best { color: #22c55e; font-weight: 600; }
.bar-wrap { background: rgba(255,255,255,0.04); border-radius: 3px; height: 5px; width: 120px; }
.bar-fill { height: 100%; border-radius: 3px; }

/* ── RESEARCH CARD ── */
.research-card {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 40px;
    display: grid; grid-template-columns: 1fr 1px 1fr 1px 1fr;
    gap: 36px; align-items: start;
}
.r-divider { background: rgba(255,255,255,0.06); }
.r-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 10px;
}
.r-h3 { font-size: 15px; font-weight: 600; color: #f1f5f9; margin-bottom: 10px; }
.r-body { font-size: 13px; color: #64748b; line-height: 1.75; }

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    background: #04070c;
    padding: 28px 48px;
    display: flex; justify-content: space-between; align-items: center;
}
.footer-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #1e293b;
}
.footer-brand strong { color: #334155; }
.footer-links {
    display: flex; gap: 20px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
}
.footer-links a { color: #334155; text-decoration: none; }
.footer-links a:hover { color: #22c55e; }

/* ── Streamlit overrides ── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; font-weight: 500 !important;
    border-radius: 6px !important; padding: 10px 28px !important;
    letter-spacing: 0.04em !important; transition: all 0.2s !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"] {
    background: #22c55e !important; color: #051a0d !important; border: none !important;
}
div.stButton > button[kind="primary"]:hover { background: #16a34a !important; }
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #475569 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important; border-color: rgba(255,255,255,0.2) !important;
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
</style>
""", unsafe_allow_html=True)

# ── NAVIGATION
st.markdown("""
<div class="top-nav">
    <div class="nav-brand">
        <span class="brand-dot"></span>
        CAML-TC
    </div>
    <div class="nav-links">
        <a class="nav-link" href="/overview">Dashboard</a>
        <a class="nav-link" href="/forecast">Forecast</a>
        <a class="nav-link" href="/simulation">Simulation</a>
        <a class="nav-link" href="https://pypi.org/project/caml-tc" target="_blank">PyPI &#8599;</a>
    </div>
    <div class="nav-pill">UK National Grid &middot; ESO API</div>
</div>
""", unsafe_allow_html=True)

# ── HERO
st.markdown("""
<div class="hero-wrap">
    <div class="hero-cite-top">
        IEEE EEEIC 2026 &nbsp;&middot;&nbsp; Scopus-indexed
        &nbsp;&middot;&nbsp; Ulster University &amp; Solent University (via QA)
    </div>
    <h1 class="hero-h1">
        The carbon cost of ML training<br>is a matter of timing
    </h1>
    <p class="hero-sub">
        CAML-TC uses real-time UK National Grid carbon data and a risk-aware Q-learning agent
        to temporally shift ML training workloads into low-carbon grid windows &mdash; reducing
        emissions by 28&ndash;34% in realistic conditions and over 50% under high grid variability.
    </p>
    <p class="hero-install">
        <code>pip install caml-tc</code> &nbsp;&middot;&nbsp; MIT License &nbsp;&middot;&nbsp; Python 3.8+
    </p>
    <div class="hero-cta">
        <a class="btn-primary" href="/overview" target="_self">Open Dashboard &rarr;</a>
        <a class="btn-secondary" href="/simulation" target="_self">Run Simulation</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── IMPACT STRIP
st.markdown("""
<div class="impact-row">
    <div class="impact-cell">
        <span class="impact-n">28&ndash;34%</span>
        <span class="impact-l">CO&#8322; reduction<br>realistic UK grid conditions</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">&gt;50%</span>
        <span class="impact-l">CO&#8322; reduction<br>high grid variability</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">8,000</span>
        <span class="impact-l">RL training episodes<br>real UK National Grid data</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">24 hr</span>
        <span class="impact-l">forecast horizon<br>exponential confidence decay</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SYSTEM ARCHITECTURE
st.markdown("""
<div class="section">
    <div class="section-eye">System Architecture</div>
    <h2 class="section-h2">From raw grid data to scheduling decision</h2>
    <p class="section-sub">A closed-loop pipeline operating on real UK National Grid ESO data at 30-minute resolution</p>
    <div class="pipeline">
        <div class="pipe-step">
            <div class="pipe-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M5 12.55a11 11 0 0 1 14.08 0"/>
                    <path d="M1.42 9a16 16 0 0 1 21.16 0"/>
                    <path d="M8.53 16.11a6 6 0 0 1 6.95 0"/>
                    <circle cx="12" cy="20" r="1" fill="#22c55e" stroke="none"/>
                </svg>
            </div>
            <div class="pipe-title">Grid API</div>
            <div class="pipe-desc">Real-time + 24 h forecast from National Grid ESO at 30-min resolution</div>
        </div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step">
            <div class="pipe-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>
                </svg>
            </div>
            <div class="pipe-title">Carbon Intelligence</div>
            <div class="pipe-desc">Exponential confidence decay &middot; peak/low detection &middot; uncertainty modelling</div>
        </div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step">
            <div class="pipe-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2"/>
                    <rect x="9" y="9" width="6" height="6"/>
                    <line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>
                    <line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>
                    <line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>
                    <line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
                </svg>
            </div>
            <div class="pipe-title">Scheduling Engine</div>
            <div class="pipe-desc">Heuristic baseline + Q-learning RL agent &middot; auto-selects strategy by grid volatility</div>
        </div>
        <div class="pipe-arrow">&#8594;</div>
        <div class="pipe-step">
            <div class="pipe-icon">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 11l3 3L22 4"/>
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                </svg>
            </div>
            <div class="pipe-title">Optimal Window</div>
            <div class="pipe-desc">Execution window with quantified CO&#8322; savings vs immediate baseline</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── KEY CONTRIBUTIONS
st.markdown("""
<div class="section" style="padding-top: 0;">
    <div class="section-eye">Key Contributions</div>
    <h2 class="section-h2">Research contributions</h2>
    <p class="section-sub">Peer-reviewed contributions covering RL reward design, uncertainty modelling, and reproducible emissions measurement</p>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                    <path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/>
                    <path d="M7 21h10"/><line x1="12" y1="3" x2="12" y2="21"/>
                    <path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>
                </svg>
            </div>
            <div class="feat-title">Multi-objective RL reward</div>
            <div class="feat-body">Jointly optimises carbon intensity, forecast uncertainty, delay penalty, and deadline constraints &mdash; not just the lowest-carbon slot.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/>
                    <polyline points="16 17 22 17 22 11"/>
                </svg>
            </div>
            <div class="feat-title">Exponential confidence decay</div>
            <div class="feat-body"><code>w&#7511; = exp(&minus;&lambda;&middot;&Delta;t)</code> &mdash; prevents over-commitment to unreliable long-horizon predictions. Forecast trust decreases monotonically with horizon distance.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="16 3 21 3 21 8"/>
                    <line x1="4" y1="20" x2="21" y2="3"/>
                    <polyline points="21 16 21 21 16 21"/>
                    <line x1="15" y1="15" x2="21" y2="21"/>
                    <line x1="4" y1="4" x2="9" y2="9"/>
                </svg>
            </div>
            <div class="feat-title">Stochastic noise injection</div>
            <div class="feat-body">Training injects real grid variability noise, forcing the RL agent to learn robust policies that generalise beyond clean-signal conditions.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="4" y="4" width="16" height="16" rx="2"/>
                    <rect x="9" y="9" width="6" height="6"/>
                    <line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>
                    <line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>
                    <line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>
                    <line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>
                </svg>
            </div>
            <div class="feat-title">Hybrid strategy selection</div>
            <div class="feat-body">Heuristic scheduler handles stable grids. RL agent activates when volatility exceeds threshold (&sigma; &gt; 30 gCO&#8322;/kWh) &mdash; where static rules break down.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M9 11l3 3L22 4"/>
                    <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>
                </svg>
            </div>
            <div class="feat-title">CodeCarbon validation</div>
            <div class="feat-body">Results validated with CodeCarbon emissions measurement &mdash; not simulated numbers. Reproducible experimental pipeline across 5 runs.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="m7.5 4.27 9 5.15"/>
                    <path d="M21 8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16Z"/>
                    <path d="m3.3 7 8.7 5 8.7-5"/>
                    <line x1="12" y1="22" x2="12" y2="12"/>
                </svg>
            </div>
            <div class="feat-title">pip-installable library</div>
            <div class="feat-body">Integrates with any existing training loop in three lines of code. No infrastructure changes, no model retraining. Compatible with PyTorch, TensorFlow, and custom loops.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── VALIDATED RESULTS
st.markdown("""
<div class="results-strip">
    <div class="results-inner">
        <div class="section-eye" style="margin-bottom:10px;">Validated Results</div>
        <h2 class="section-h2">Tested on real UK National Grid data</h2>
        <p class="section-sub">Emissions measured with CodeCarbon across 5 experimental runs</p>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Conditions</th>
                    <th>CO&#8322; Reduction</th>
                    <th>Relative Emissions</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="td-strategy" style="color:#ef4444;">Baseline</td>
                    <td>Immediate execution</td>
                    <td style="color:#ef4444;">0%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:100%;background:#ef4444;"></div></div></td>
                    <td>Industry default &mdash; no carbon awareness</td>
                </tr>
                <tr>
                    <td class="td-strategy" style="color:#f59e0b;">Heuristic</td>
                    <td>Realistic UK grid</td>
                    <td style="color:#f59e0b;">~28%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:72%;background:#f59e0b;"></div></div></td>
                    <td>Strong interpretable baseline</td>
                </tr>
                <tr>
                    <td class="td-strategy">RL Agent (CAML-TC)</td>
                    <td>Realistic UK grid</td>
                    <td class="td-best">28&ndash;34%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:66%;background:#22c55e;"></div></div></td>
                    <td>Matches heuristic on stable grids</td>
                </tr>
                <tr>
                    <td class="td-strategy">RL Agent (CAML-TC)</td>
                    <td>High volatility / optimal</td>
                    <td class="td-best">&gt;50%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:48%;background:#22c55e;"></div></div></td>
                    <td>RL outperforms heuristic where it matters most</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
""", unsafe_allow_html=True)

# ── RESEARCH CONTEXT
st.markdown("""
<div class="section">
    <div class="section-eye">Research Context</div>
    <h2 class="section-h2">Bridging the gap between industry and research</h2>
    <div class="research-card" style="margin-top:24px;">
        <div>
            <div class="r-label">The Problem</div>
            <h3 class="r-h3">Proprietary and inaccessible</h3>
            <p class="r-body">Google has run carbon-aware compute shifting internally since 2020. Microsoft released a partial SDK. Neither is available to independent researchers, university labs, or individual engineers &mdash; the people responsible for a rapidly growing share of AI compute.</p>
        </div>
        <div class="r-divider"></div>
        <div>
            <div class="r-label">The Approach</div>
            <h3 class="r-h3">Open, deployable, reproducible</h3>
            <p class="r-body">CAML-TC is a lightweight RL-enhanced carbon optimisation layer that runs on top of any existing training loop. It uses the UK government&rsquo;s free public Carbon Intensity API &mdash; infrastructure already built for exactly this purpose.</p>
        </div>
        <div class="r-divider"></div>
        <div>
            <div class="r-label">UK Net Zero Alignment</div>
            <h3 class="r-h3">Data centres are a growing problem</h3>
            <p class="r-body">UK data centres and AI infrastructure are among the fastest-growing electricity consumers. CAML-TC provides a scheduling intelligence layer missing from the open-source ecosystem, making carbon-aware ML accessible without changing model code.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER
st.markdown("""
<div class="site-footer">
    <div class="footer-brand">
        <strong>CAML-TC</strong> &middot; Carbon-Aware ML Training Controller &middot;
        Sufiyan Ul Rehman &middot; Ulster University &amp; Solent University (via QA)
    </div>
    <div class="footer-links">
        <a href="https://pypi.org/project/caml-tc" target="_blank">PyPI</a>
        <a href="https://github.com/sufirehman/carbon-aware-ml-scheduler" target="_blank">GitHub</a>
        <a href="https://ieeexplore.ieee.org" target="_blank">IEEE EEEIC 2026</a>
        <span style="color:#1e293b;">MIT License &middot; 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)
