import streamlit as st

st.set_page_config(
    page_title="CAML-TC · Carbon-Aware ML Scheduler",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# SHARED STYLES
# ══════════════════════════════════════════════════════════════
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

/* ── GRID TEXTURE ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,197,94,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.03) 1px, transparent 1px);
    background-size: 52px 52px;
}

/* ── LIVE TICKER ── */
.ticker-bar {
    position: relative; z-index: 10;
    background: #0a1018;
    border-bottom: 1px solid rgba(34,197,94,0.12);
    padding: 9px 0;
    overflow: hidden;
}
.ticker-track {
    display: inline-flex; gap: 56px;
    animation: scroll 35s linear infinite;
    white-space: nowrap;
}
@keyframes scroll { from { transform: translateX(0) } to { transform: translateX(-50%) } }
.t-item {
    display: inline-flex; align-items: center; gap: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10.5px; letter-spacing: 0.06em;
}
.t-key   { color: #334155; text-transform: uppercase; }
.t-val   { color: #94a3b8; font-weight: 500; }
.t-up    { color: #22c55e; }
.t-dn    { color: #ef4444; }
.t-pipe  { color: #1e293b; }

/* ── TOP NAV ── */
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
    width: 9px; height: 9px; border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 12px rgba(34,197,94,0.7);
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.6;transform:scale(0.85)} }
.nav-links {
    display: flex; gap: 32px;
    font-size: 12px; color: #475569;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}
.nav-link {
    color: #475569; text-decoration: none;
    transition: color 0.2s; letter-spacing: 0.05em;
}
.nav-link:hover { color: #22c55e; }
.nav-pill {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #22c55e;
    background: rgba(34,197,94,0.08);
    border: 1px solid rgba(34,197,94,0.22);
    padding: 6px 16px; border-radius: 20px;
    display: flex; align-items: center; gap: 7px;
    letter-spacing: 0.06em;
}

/* ── HERO ── */
.hero-wrap {
    max-width: 900px; margin: 0 auto;
    padding: 96px 32px 72px;
    text-align: center;
    position: relative; z-index: 1;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #22c55e;
    background: rgba(34,197,94,0.07);
    border: 1px solid rgba(34,197,94,0.18);
    padding: 6px 18px; border-radius: 20px;
    letter-spacing: 0.1em; text-transform: uppercase;
    margin-bottom: 32px;
}
.hero-h1 {
    font-size: 62px; font-weight: 700;
    line-height: 1.08; letter-spacing: -0.03em;
    color: #f1f5f9; margin-bottom: 24px;
}
.hero-h1 .accent { color: #22c55e; }
.hero-sub {
    font-size: 18px; font-weight: 300;
    color: #64748b; line-height: 1.75;
    max-width: 620px; margin: 0 auto 16px;
}
.hero-cite {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #334155;
    letter-spacing: 0.06em; margin-bottom: 48px;
}
.hero-cite a { color: #22c55e; text-decoration: none; }

/* ── IMPACT ROW ── */
.impact-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    background: rgba(255,255,255,0.012);
    position: relative; z-index: 1;
}
.impact-cell {
    padding: 44px 20px; text-align: center;
    border-right: 1px solid rgba(255,255,255,0.06);
    transition: background 0.25s;
}
.impact-cell:last-child { border-right: none; }
.impact-cell:hover { background: rgba(34,197,94,0.035); }
.impact-n {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 42px; font-weight: 600;
    color: #22c55e; display: block; margin-bottom: 8px;
    letter-spacing: -0.02em;
    text-shadow: 0 0 28px rgba(34,197,94,0.25);
}
.impact-l {
    font-size: 11px; color: #475569;
    text-transform: uppercase; letter-spacing: 0.07em;
    line-height: 1.55;
}

/* ── HOW IT WORKS ── */
.section { max-width: 1080px; margin: 0 auto; padding: 72px 32px; position: relative; z-index: 1; }
.section-eye {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.14em;
    margin-bottom: 12px;
    display: flex; align-items: center; gap: 10px;
}
.section-eye::before { content:''; display:inline-block; width:24px; height:1px; background:#22c55e; }
.section-h2 {
    font-size: 34px; font-weight: 600;
    color: #f1f5f9; letter-spacing: -0.015em; margin-bottom: 8px;
}
.section-sub { font-size: 14px; color: #64748b; margin-bottom: 40px; }

/* ── PIPELINE ── */
.pipeline {
    display: grid; grid-template-columns: 1fr 28px 1fr 28px 1fr 28px 1fr;
    align-items: center; gap: 0;
    margin: 48px 0;
}
.pipe-step {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 28px 20px;
    text-align: center; transition: all 0.25s;
}
.pipe-step:hover { border-color: rgba(34,197,94,0.25); background: #111e2d; }
.pipe-icon { font-size: 26px; margin-bottom: 12px; }
.pipe-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #94a3b8;
    font-weight: 500; margin-bottom: 6px;
    text-transform: uppercase; letter-spacing: 0.06em;
}
.pipe-desc { font-size: 12px; color: #475569; line-height: 1.6; }
.pipe-arrow { text-align: center; color: #1e3a2a; font-size: 18px; }

/* ── FEATURE CARDS ── */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
.feat-card {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 30px 24px;
    transition: all 0.25s; position: relative; overflow: hidden;
}
.feat-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: transparent; transition: background 0.25s;
}
.feat-card:hover { border-color: rgba(34,197,94,0.2); transform: translateY(-3px); }
.feat-card:hover::after { background: #22c55e; }
.feat-icon { font-size: 22px; margin-bottom: 14px; }
.feat-title { font-size: 16px; font-weight: 600; color: #e2e8f0; margin-bottom: 10px; }
.feat-body { font-size: 13px; color: #64748b; line-height: 1.7; }

/* ── RESULTS STRIP ── */
.results-strip {
    background: #0a1018;
    border-top: 1px solid rgba(255,255,255,0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
    padding: 64px 0;
}
.results-inner { max-width: 1080px; margin: 0 auto; padding: 0 32px; }
.results-table {
    width: 100%; border-collapse: collapse;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 32px;
}
.results-table th {
    font-size: 10px; color: #334155;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0 20px 14px; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.results-table td {
    font-size: 13px; color: #94a3b8;
    padding: 16px 20px;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.results-table tr:last-child td { border-bottom: none; }
.td-strategy { color: #f1f5f9; font-weight: 500; }
.td-best { color: #22c55e; font-weight: 600; }
.bar-wrap { background: rgba(255,255,255,0.04); border-radius: 3px; height: 6px; width: 140px; }
.bar-fill { height: 100%; border-radius: 3px; }

/* ── RESEARCH BLOCK ── */
.research-card {
    background: #0c1420;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 48px;
    display: grid; grid-template-columns: 1fr 1px 1fr 1px 1fr;
    gap: 40px; align-items: start;
}
.r-divider { background: rgba(255,255,255,0.06); }
.r-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 12px;
}
.r-h3 { font-size: 17px; font-weight: 600; color: #f1f5f9; margin-bottom: 10px; }
.r-body { font-size: 13px; color: #64748b; line-height: 1.75; }

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid rgba(255,255,255,0.06);
    background: #04070c;
    padding: 32px 48px;
    display: flex; justify-content: space-between; align-items: center;
}
.footer-brand {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #1e293b;
}
.footer-brand strong { color: #334155; }
.footer-links {
    display: flex; gap: 24px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #1e293b;
}
.footer-links a { color: #334155; text-decoration: none; }
.footer-links a:hover { color: #22c55e; }

/* ── HERO CTA BUTTONS ── */
.hero-cta {
    display: flex; align-items: center; justify-content: center;
    gap: 14px; margin-top: 40px;
}
.btn-primary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 500; letter-spacing: 0.04em;
    color: #051a0d; background: #22c55e;
    border: none; border-radius: 8px;
    padding: 12px 28px; text-decoration: none;
    transition: background 0.2s, transform 0.15s;
    display: inline-block;
}
.btn-primary:hover { background: #16a34a; transform: translateY(-1px); }
.btn-secondary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 400; letter-spacing: 0.04em;
    color: #22c55e; background: rgba(34,197,94,0.06);
    border: 1px solid rgba(34,197,94,0.28); border-radius: 8px;
    padding: 12px 28px; text-decoration: none;
    transition: color 0.2s, border-color 0.2s, background 0.2s;
    display: inline-block;
}
.btn-secondary:hover { background: rgba(34,197,94,0.12); border-color: rgba(34,197,94,0.5); }

/* ── STREAMLIT BUTTON OVERRIDE (other pages) ── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; font-weight: 500 !important;
    border-radius: 8px !important; padding: 10px 28px !important;
    letter-spacing: 0.04em !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
div.stButton > button[kind="primary"] {
    background: #22c55e !important; color: #051a0d !important;
    border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #16a34a !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #475569 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important;
    border-color: rgba(255,255,255,0.2) !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# LIVE TICKER  (static values — replace with API call if needed)
# ══════════════════════════════════════════════════════════════
ticker_items = [
    ("CARBON NOW",      "178 gCO₂/kWh", "▼ 12%",  "up"),
    ("WIND CAPACITY",   "32.4 GW",       "▲ 4.1%", "up"),
    ("SOLAR OUTPUT",    "7.1 GW",        "▲ 2.2%", "up"),
    ("GRID DEMAND",     "38.2 GW",       "▼ 1.0%", "dn"),
    ("COAL ONLINE",     "0 GW",          "NET ZERO","up"),
    ("RENEWABLES MIX",  "58%",           "▲ 6pt",  "up"),
    ("FORECAST HORIZON","24 hr",         "LIVE",    "up"),
]
items_html = ""
for label, val, delta, d in ticker_items * 2:
    cls = "t-up" if d == "up" else "t-dn"
    items_html += f"""
    <span class="t-item">
        <span class="t-key">{label}</span>
        <span class="t-val">{val}</span>
        <span class="{cls}">{delta}</span>
        <span class="t-pipe">|</span>
    </span>"""
st.markdown(f'<div class="ticker-bar"><div class="ticker-track">{items_html}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TOP NAV
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="top-nav">
    <div class="nav-brand"><span class="brand-dot"></span> CAML-TC</div>
    <div class="nav-links">
        <a class="nav-link" href="/overview">Dashboard</a>
        <a class="nav-link" href="/forecast">Forecast</a>
        <a class="nav-link" href="/simulation">Simulation</a>
        <a class="nav-link" href="https://pypi.org/project/caml-tc" target="_blank">PyPI &nearr;</a>
    </div>
    <div class="nav-pill"><span class="brand-dot"></span> UK NATIONAL GRID &middot; LIVE</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════
st.markdown("""<div class="hero-wrap">
    <div class="hero-eyebrow"><span class="brand-dot"></span> IEEE EEEIC 2026 &middot; Peer-Reviewed Research</div>
    <h1 class="hero-h1">
        The carbon cost of AI<br>is <span class="accent">a matter of timing</span>
    </h1>
    <p class="hero-sub">
        CAML-TC uses real-time UK National Grid data and a risk-aware reinforcement learning agent 
        to shift ML training into low-carbon windows &mdash; cutting emissions by up to 34% without 
        touching a single line of model code.
    </p>
    <p class="hero-cite">
        Published at <a href="https://ieeexplore.ieee.org" target="_blank">IEEE EEEIC 2026</a> &middot;
        Q1, Scopus-indexed &middot; <code>pip install caml-tc</code>
    </p>
    <div class="hero-cta">
        <a class="btn-primary" href="/overview" target="_self">Open Dashboard &rarr;</a>
        <a class="btn-secondary" href="/simulation" target="_self">Run Simulation Lab</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# IMPACT NUMBERS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="impact-row">
    <div class="impact-cell">
        <span class="impact-n">↓ 34%</span>
        <span class="impact-l">CO₂ reduction<br>realistic conditions</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">&gt; 50%</span>
        <span class="impact-l">CO₂ reduction<br>optimal low-variability</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">8,000</span>
        <span class="impact-l">RL training episodes<br>on real UK grid data</span>
    </div>
    <div class="impact-cell">
        <span class="impact-n">24 hr</span>
        <span class="impact-l">forecast horizon<br>with uncertainty decay</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# HOW IT WORKS — PIPELINE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-eye">System Architecture</div>
    <h2 class="section-h2">From raw grid data to scheduling decision</h2>
    <p class="section-sub">A closed-loop pipeline — no manual intervention required</p>
    <div class="pipeline">
        <div class="pipe-step">
            <div class="pipe-icon">📡</div>
            <div class="pipe-title">Grid API</div>
            <div class="pipe-desc">Real-time + 24h forecast from UK National Grid ESO at 30-min resolution</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">🧮</div>
            <div class="pipe-title">Carbon Intelligence</div>
            <div class="pipe-desc">Exponential confidence decay · peak/low detection · uncertainty modelling</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">🤖</div>
            <div class="pipe-title">Scheduling Engine</div>
            <div class="pipe-desc">Heuristic baseline + Q-learning RL agent · automatically selects best strategy</div>
        </div>
        <div class="pipe-arrow">→</div>
        <div class="pipe-step">
            <div class="pipe-icon">✅</div>
            <div class="pipe-title">Optimal Window</div>
            <div class="pipe-desc">Execution window with quantified CO₂ savings vs immediate baseline</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FEATURE GRID
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section" style="padding-top: 0;">
    <div class="section-eye">Key Innovations</div>
    <h2 class="section-h2">What makes CAML-TC different</h2>
    <p class="section-sub">Built on peer-reviewed ML research, not just heuristics</p>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">⚖️</div>
            <div class="feat-title">Multi-objective RL reward</div>
            <div class="feat-body">Jointly optimises carbon intensity, forecast uncertainty, delay penalty, and deadline constraints — not just the lowest carbon slot.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📉</div>
            <div class="feat-title">Exponential confidence decay</div>
            <div class="feat-body"><code>wₜ = exp(−λ·Δt)</code> — prevents over-commitment to unreliable long-horizon predictions. The further ahead, the less the agent trusts the forecast.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎲</div>
            <div class="feat-title">Stochastic noise injection</div>
            <div class="feat-body">Training injects real grid variability noise, forcing the RL agent to learn robust policies — not just patterns that work on clean signals.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Hybrid strategy selection</div>
            <div class="feat-body">Heuristic scheduler handles stable grids. RL agent activates when volatility exceeds threshold — where static rules break down and learning matters most.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🧪</div>
            <div class="feat-title">CodeCarbon validation</div>
            <div class="feat-body">Results validated with CodeCarbon emissions measurement — not simulated numbers. Reproducible, peer-reviewed experimental pipeline.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📦</div>
            <div class="feat-title">pip-installable library</div>
            <div class="feat-body">Drop into any existing ML pipeline in three lines of code. No infrastructure changes. No model retraining. Works with PyTorch, TensorFlow, and custom loops.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# RESULTS TABLE
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="results-strip">
    <div class="results-inner">
        <div class="section-eye" style="margin-bottom:12px;">Validated Results</div>
        <h2 class="section-h2">Tested on real UK National Grid data</h2>
        <p class="section-sub">Emissions measured with CodeCarbon across 5 experimental runs</p>
        <table class="results-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Conditions</th>
                    <th>CO₂ Reduction</th>
                    <th>Relative Performance</th>
                    <th>Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="td-strategy" style="color:#ef4444;">Baseline</td>
                    <td>Immediate execution</td>
                    <td style="color:#ef4444;">0%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:100%;background:#ef4444;"></div></div></td>
                    <td>Industry default — no carbon awareness</td>
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
                    <td class="td-best">28 – 34%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:66%;background:#22c55e;"></div></div></td>
                    <td>Matches heuristic on stable grids</td>
                </tr>
                <tr>
                    <td class="td-strategy">RL Agent (CAML-TC)</td>
                    <td>High volatility / optimal</td>
                    <td class="td-best">> 50%</td>
                    <td><div class="bar-wrap"><div class="bar-fill" style="width:48%;background:#22c55e;"></div></div></td>
                    <td>RL outperforms heuristic where it matters most</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# RESEARCH CONTEXT
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="section">
    <div class="section-eye">Research Context</div>
    <h2 class="section-h2">Bridging the gap between industry and research</h2>
    <div class="research-card" style="margin-top:28px;">
        <div>
            <div class="r-label">The Problem</div>
            <h3 class="r-h3">Proprietary and inaccessible</h3>
            <p class="r-body">Google has run carbon-aware compute shifting internally since 2020. Microsoft released a partial SDK. Neither is available to independent researchers, university labs, or individual engineers — the people responsible for a rapidly growing share of AI compute.</p>
        </div>
        <div class="r-divider"></div>
        <div>
            <div class="r-label">The Approach</div>
            <h3 class="r-h3">Open, deployable, reproducible</h3>
            <p class="r-body">CAML-TC is a lightweight RL-enhanced carbon optimisation layer that runs on top of any existing training loop. It uses the UK government's own free public carbon intensity API — infrastructure already built for exactly this purpose.</p>
        </div>
        <div class="r-divider"></div>
        <div>
            <div class="r-label">UK Net Zero Alignment</div>
            <h3 class="r-h3">Data centres are a growing problem</h3>
            <p class="r-body">UK data centres and AI infrastructure are among the fastest-growing electricity consumers. CAML-TC provides the scheduling intelligence layer that was missing — making carbon-aware ML accessible to anyone, without changing a single line of model code.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="site-footer">
    <div class="footer-brand">
        <strong>CAML-TC</strong> · Carbon-Aware ML Training Controller · 
        Sufiyan Ul Rehman · Ulster University &amp; Solent University (via QA)
    </div>
    <div class="footer-links">
        <a href="https://pypi.org/project/caml-tc" target="_blank">PyPI</a>
        <a href="https://github.com/sufirehman/carbon-aware-ml-scheduler" target="_blank">GitHub</a>
        <a href="https://ieeexplore.ieee.org" target="_blank">IEEE Paper</a>
        <span>MIT License · 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)