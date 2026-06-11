import streamlit as st

st.set_page_config(
    page_title="CAML-TC · Carbon-Aware ML Scheduler",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, .stApp {
    background: #04080f !important;
    font-family: 'Inter', sans-serif;
    color: #c8d6e5;
    overflow-x: hidden;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

:root {
    --green: #16c964;
    --green-dim: #0f9e4a;
    --surface: #0b1220;
    --surface2: #111928;
    --border: rgba(255,255,255,0.07);
    --muted: #5a6a7e;
    --bright: #e8f0f8;
}

/* ── GRID TEXTURE ── */
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(22,201,100,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(22,201,100,0.025) 1px, transparent 1px);
    background-size: 52px 52px;
}

/* ── NAV ── */
.top-nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 40px;
    border-bottom: 1px solid var(--border);
    position: sticky; top: 0;
    background: rgba(4,8,15,0.97);
    z-index: 100;
}
.nav-logo {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; font-weight: 500;
    color: var(--bright); letter-spacing: 0.06em;
    display: flex; align-items: center; gap: 8px;
}
.nav-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); flex-shrink: 0;
    animation: blink 2.5s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }
.nav-links {
    display: flex; align-items: center; gap: 28px;
}
.nav-links a {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px; color: var(--muted);
    text-decoration: none; letter-spacing: 0.04em;
    transition: color 0.18s;
}
.nav-links a:hover { color: var(--green); }
.nav-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--green);
    border: 1px solid rgba(22,201,100,0.28);
    padding: 5px 14px; border-radius: 6px;
    letter-spacing: 0.06em;
}

/* ── HERO ── */
.hero {
    max-width: 780px; margin: 0 auto;
    padding: 88px 32px 72px; text-align: center;
    position: relative; z-index: 1;
}
.hero-tag {
    display: inline-flex; align-items: center; gap: 7px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: var(--green);
    letter-spacing: 0.1em; text-transform: uppercase;
    background: rgba(22,201,100,0.06);
    border: 1px solid rgba(22,201,100,0.2);
    padding: 5px 16px; border-radius: 20px;
    margin-bottom: 28px;
}
.hero h1 {
    font-size: clamp(36px, 6vw, 62px);
    font-weight: 700; line-height: 1.07;
    letter-spacing: -0.03em;
    color: var(--bright); margin-bottom: 20px;
}
.hero h1 em { font-style: normal; color: var(--green); }
.hero-sub {
    font-size: clamp(15px, 2vw, 18px);
    font-weight: 300; color: var(--muted);
    line-height: 1.75; max-width: 560px;
    margin: 0 auto 12px;
}
.hero-meta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #2a3d52;
    letter-spacing: 0.05em; margin-bottom: 40px;
}
.hero-meta a { color: var(--green); text-decoration: none; }
.hero-meta code {
    background: rgba(22,201,100,0.07);
    border: 1px solid rgba(22,201,100,0.15);
    padding: 2px 8px; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #7ddba8;
}

/* ── CTA BUTTONS ── */
.cta-row {
    display: flex; align-items: center;
    justify-content: center; gap: 12px;
    flex-wrap: wrap;
}
.btn-p {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; font-weight: 500; letter-spacing: 0.04em;
    color: #03140a; background: var(--green);
    border: none; border-radius: 8px;
    padding: 13px 28px; text-decoration: none;
    display: inline-block; transition: background 0.18s;
    cursor: pointer;
}
.btn-p:hover { background: var(--green-dim); }
.btn-s {
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px; font-weight: 400; letter-spacing: 0.04em;
    color: var(--green); background: rgba(22,201,100,0.06);
    border: 1px solid rgba(22,201,100,0.25); border-radius: 8px;
    padding: 13px 28px; text-decoration: none;
    display: inline-block; transition: all 0.18s;
    cursor: pointer;
}
.btn-s:hover { background: rgba(22,201,100,0.12); border-color: rgba(22,201,100,0.45); }

/* ── STATS BAR ── */
.stats-bar {
    display: grid; grid-template-columns: repeat(4, 1fr);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    position: relative; z-index: 1;
}
.stat-cell {
    padding: 36px 20px; text-align: center;
    border-right: 1px solid var(--border);
    transition: background 0.2s;
}
.stat-cell:last-child { border-right: none; }
.stat-cell:hover { background: rgba(22,201,100,0.03); }
.stat-n {
    font-family: 'JetBrains Mono', monospace;
    font-size: clamp(28px, 4vw, 40px); font-weight: 600;
    color: var(--green); display: block; margin-bottom: 6px;
    letter-spacing: -0.02em;
}
.stat-l {
    font-size: 11px; color: var(--muted);
    text-transform: uppercase; letter-spacing: 0.08em; line-height: 1.55;
}

/* ── SECTIONS ── */
.section {
    max-width: 1060px; margin: 0 auto;
    padding: 72px 32px; position: relative; z-index: 1;
}
.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--green);
    text-transform: uppercase; letter-spacing: 0.16em; margin-bottom: 10px;
}
.section h2 {
    font-size: clamp(22px, 3.5vw, 32px); font-weight: 600;
    color: var(--bright); letter-spacing: -0.02em; margin-bottom: 8px;
}
.section-sub { font-size: 14px; color: var(--muted); margin-bottom: 40px; }

/* ── PIPELINE ── */
.pipeline {
    display: grid;
    grid-template-columns: 1fr auto 1fr auto 1fr auto 1fr;
    align-items: center; margin-top: 40px;
}
.pipe-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px; padding: 24px 18px; text-align: center;
    transition: border-color 0.2s, background 0.2s;
}
.pipe-box:hover { border-color: rgba(22,201,100,0.22); background: var(--surface2); }
.pipe-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--green);
    letter-spacing: 0.1em; margin-bottom: 8px;
}
.pipe-name { font-size: 13px; font-weight: 600; color: var(--bright); margin-bottom: 6px; }
.pipe-desc { font-size: 12px; color: var(--muted); line-height: 1.55; }
.pipe-arr { color: #1a3a2a; font-size: 16px; padding: 0 6px; text-align: center; }

/* ── FEATURE GRID ── */
.feat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; }
.feat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 26px 22px;
    position: relative; overflow: hidden; transition: border-color 0.2s;
}
.feat-card::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
    background: transparent; transition: background 0.2s;
}
.feat-card:hover { border-color: rgba(22,201,100,0.18); }
.feat-card:hover::after { background: var(--green); }
.feat-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--green);
    text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 8px;
}
.feat-title { font-size: 15px; font-weight: 600; color: var(--bright); margin-bottom: 8px; }
.feat-body { font-size: 13px; color: var(--muted); line-height: 1.65; }
.feat-body code {
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    background: rgba(22,201,100,0.07); padding: 1px 6px; border-radius: 3px; color: #7ddba8;
}

/* ── RESULTS ── */
.results-bg {
    background: var(--surface);
    border-top: 1px solid var(--border);
    border-bottom: 1px solid var(--border);
    padding: 64px 0; position: relative; z-index: 1;
}
.results-inner { max-width: 1060px; margin: 0 auto; padding: 0 32px; }
.res-table {
    width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    margin-top: 32px;
}
.res-table th {
    font-size: 10px; color: #2a3d52;
    text-transform: uppercase; letter-spacing: 0.1em;
    padding: 0 16px 14px; text-align: left;
    border-bottom: 1px solid var(--border);
}
.res-table td {
    font-size: 12px; color: #6a7f94;
    padding: 14px 16px;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.res-table tr:last-child td { border-bottom: none; }
.td-name { color: var(--bright); font-weight: 500; }
.td-g { color: var(--green); font-weight: 600; }
.td-r { color: #d95555; }
.td-a { color: #c49a14; }
.bar-bg { background: rgba(255,255,255,0.05); border-radius: 3px; height: 5px; width: 120px; }
.bar-fill { height: 100%; border-radius: 3px; }

/* ── RESEARCH ── */
.research-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: 12px; overflow: hidden;
    margin-top: 28px;
}
.r-cell { background: #04080f; padding: 36px 30px; }
.r-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 10px; color: var(--green);
    text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px;
}
.r-title { font-size: 16px; font-weight: 600; color: var(--bright); margin-bottom: 10px; }
.r-body { font-size: 13px; color: var(--muted); line-height: 1.75; }

/* ── FOOTER ── */
.site-footer {
    border-top: 1px solid var(--border);
    background: #03060c;
    padding: 28px 40px;
    display: flex; justify-content: space-between;
    align-items: center; flex-wrap: wrap; gap: 16px;
}
.footer-l {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #213040;
}
.footer-l strong { color: #334d60; }
.footer-r { display: flex; gap: 20px; flex-wrap: wrap; }
.footer-r a {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px; color: #213040;
    text-decoration: none; transition: color 0.18s;
}
.footer-r a:hover { color: var(--green); }

/* ── MOBILE ── */
@media(max-width:720px) {
    .top-nav { padding: 14px 20px; }
    .nav-links { display: none; }
    .hero { padding: 56px 20px 48px; }
    .stats-bar { grid-template-columns: repeat(2, 1fr); }
    .stat-cell:nth-child(2) { border-right: none; }
    .stat-cell:nth-child(3),
    .stat-cell:nth-child(4) { border-top: 1px solid var(--border); }
    .stat-cell:nth-child(4) { border-right: none; }
    .section { padding: 48px 20px; }
    .pipeline { grid-template-columns: 1fr; gap: 12px; }
    .pipe-arr { transform: rotate(90deg); margin: 0 auto; }
    .feat-grid { grid-template-columns: 1fr; }
    .research-grid { grid-template-columns: 1fr; }
    .res-table th, .res-table td { padding: 10px 8px; font-size: 11px; }
    .bar-bg { width: 60px; }
    .results-inner { padding: 0 20px; }
    .site-footer { padding: 24px 20px; flex-direction: column; align-items: flex-start; }
}
@media(max-width:480px) {
    .cta-row { flex-direction: column; align-items: stretch; }
    .btn-p, .btn-s { text-align: center; }
}

/* Hide default streamlit padding around html components */
.stMarkdown { line-height: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── NAV
st.markdown("""
<div class="top-nav">
    <div class="nav-logo"><span class="nav-dot"></span>CAML-TC</div>
    <div class="nav-links">
        <a href="/overview">Dashboard</a>
        <a href="/forecast">Forecast</a>
        <a href="/simulation">Simulation</a>
        <a href="https://pypi.org/project/caml-tc" target="_blank">PyPI &#8599;</a>
    </div>
    <div class="nav-badge">UK GRID &middot; LIVE</div>
</div>
""", unsafe_allow_html=True)

# ── HERO
st.markdown("""
<div class="hero">
    <div class="hero-tag"><span class="nav-dot"></span>IEEE EEEIC 2026 &middot; Peer-Reviewed Research</div>
    <h1>The carbon cost of AI<br>is <em>a matter of timing</em></h1>
    <p class="hero-sub">
        CAML-TC shifts ML training into low-carbon grid windows using real-time UK National Grid
        data and a risk-aware RL agent &mdash; cutting emissions by up to 34% without changing
        a single line of model code.
    </p>
    <p class="hero-meta">
        Published at <a href="https://ieeexplore.ieee.org" target="_blank">IEEE EEEIC 2026</a>
        &middot; Q1, Scopus-indexed &middot; <code>pip install caml-tc</code>
    </p>
    <div class="cta-row">
        <a class="btn-p" href="/overview" target="_self">Open Dashboard &rarr;</a>
        <a class="btn-s" href="/simulation" target="_self">Run Simulation Lab</a>
    </div>
</div>
""", unsafe_allow_html=True)

# ── STATS BAR
st.markdown("""
<div class="stats-bar">
    <div class="stat-cell">
        <span class="stat-n">&#8595; 34%</span>
        <span class="stat-l">CO&#8322; reduction<br>realistic conditions</span>
    </div>
    <div class="stat-cell">
        <span class="stat-n">&gt; 50%</span>
        <span class="stat-l">CO&#8322; reduction<br>optimal conditions</span>
    </div>
    <div class="stat-cell">
        <span class="stat-n">8,000</span>
        <span class="stat-l">RL training episodes<br>real UK grid data</span>
    </div>
    <div class="stat-cell">
        <span class="stat-n">24 hr</span>
        <span class="stat-l">forecast horizon<br>uncertainty-weighted</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── PIPELINE
st.markdown("""
<div class="section">
    <div class="eyebrow">System Architecture</div>
    <h2>From grid data to scheduling decision</h2>
    <p class="section-sub">A closed-loop pipeline &mdash; no manual intervention required</p>
    <div class="pipeline">
        <div class="pipe-box">
            <div class="pipe-num">01</div>
            <div class="pipe-name">Grid API</div>
            <div class="pipe-desc">Real-time + 24h forecast &middot; National Grid ESO &middot; 30-min resolution</div>
        </div>
        <div class="pipe-arr">&rarr;</div>
        <div class="pipe-box">
            <div class="pipe-num">02</div>
            <div class="pipe-name">Carbon Intelligence</div>
            <div class="pipe-desc">Confidence decay &middot; peak/low detection &middot; uncertainty modelling</div>
        </div>
        <div class="pipe-arr">&rarr;</div>
        <div class="pipe-box">
            <div class="pipe-num">03</div>
            <div class="pipe-name">Scheduling Engine</div>
            <div class="pipe-desc">Heuristic baseline + Q-learning RL &middot; auto-selects best strategy</div>
        </div>
        <div class="pipe-arr">&rarr;</div>
        <div class="pipe-box">
            <div class="pipe-num">04</div>
            <div class="pipe-name">Optimal Window</div>
            <div class="pipe-desc">Execution time + quantified CO&#8322; savings vs immediate baseline</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FEATURES
st.markdown("""
<div class="section" style="padding-top:0">
    <div class="eyebrow">Key Innovations</div>
    <h2>What makes CAML-TC different</h2>
    <p class="section-sub">Peer-reviewed research &mdash; not just heuristics</p>
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-tag">RL Design</div>
            <div class="feat-title">Multi-objective reward function</div>
            <div class="feat-body">Jointly optimises carbon intensity, forecast uncertainty, delay penalty, and deadline constraints &mdash; not just the lowest carbon slot.</div>
        </div>
        <div class="feat-card">
            <div class="feat-tag">Forecasting</div>
            <div class="feat-title">Exponential confidence decay</div>
            <div class="feat-body"><code>w&#8345; = exp(&minus;&lambda;&middot;&Delta;t)</code> &mdash; prevents over-commitment to unreliable long-horizon predictions. The further ahead, the less the agent trusts the forecast.</div>
        </div>
        <div class="feat-card">
            <div class="feat-tag">Robustness</div>
            <div class="feat-title">Stochastic noise injection</div>
            <div class="feat-body">Training injects real grid variability, forcing the RL agent to learn policies robust to real-world forecast uncertainty &mdash; not just clean signals.</div>
        </div>
        <div class="feat-card">
            <div class="feat-tag">Strategy</div>
            <div class="feat-title">Hybrid scheduler selection</div>
            <div class="feat-body">Heuristic handles stable grids. RL activates above the volatility threshold &mdash; where static rules fail and adaptive learning matters most.</div>
        </div>
        <div class="feat-card">
            <div class="feat-tag">Validation</div>
            <div class="feat-title">CodeCarbon emissions measurement</div>
            <div class="feat-body">Results validated with CodeCarbon &mdash; not simulated numbers. Reproducible, peer-reviewed experimental pipeline across 5 runs.</div>
        </div>
        <div class="feat-card">
            <div class="feat-tag">Integration</div>
            <div class="feat-title">pip-installable library</div>
            <div class="feat-body">Three lines of code. No infrastructure changes. No model retraining. Works with PyTorch, TensorFlow, and custom training loops.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── RESULTS
st.markdown("""
<div class="results-bg">
    <div class="results-inner">
        <div class="eyebrow">Validated Results</div>
        <h2>Tested on real UK National Grid data</h2>
        <p class="section-sub">Emissions measured with CodeCarbon &middot; 5 experimental runs</p>
        <table class="res-table">
            <thead>
                <tr>
                    <th>Strategy</th><th>Conditions</th>
                    <th>CO&#8322; Reduction</th><th>Relative</th><th>Notes</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td class="td-name td-r">Baseline</td>
                    <td>Immediate execution</td>
                    <td class="td-r">0%</td>
                    <td><div class="bar-bg"><div class="bar-fill" style="width:100%;background:#d95555"></div></div></td>
                    <td>No carbon awareness</td>
                </tr>
                <tr>
                    <td class="td-name td-a">Heuristic</td>
                    <td>Realistic UK grid</td>
                    <td class="td-a">~28%</td>
                    <td><div class="bar-bg"><div class="bar-fill" style="width:72%;background:#c49a14"></div></div></td>
                    <td>Interpretable baseline</td>
                </tr>
                <tr>
                    <td class="td-name td-g">RL Agent (CAML-TC)</td>
                    <td>Realistic UK grid</td>
                    <td class="td-g">28&ndash;34%</td>
                    <td><div class="bar-bg"><div class="bar-fill" style="width:66%;background:#16c964"></div></div></td>
                    <td>Matches heuristic on stable grids</td>
                </tr>
                <tr>
                    <td class="td-name td-g">RL Agent (CAML-TC)</td>
                    <td>High volatility</td>
                    <td class="td-g">&gt; 50%</td>
                    <td><div class="bar-bg"><div class="bar-fill" style="width:48%;background:#16c964"></div></div></td>
                    <td>Outperforms heuristic where it matters</td>
                </tr>
            </tbody>
        </table>
    </div>
</div>
""", unsafe_allow_html=True)

# ── RESEARCH CONTEXT
st.markdown("""
<div class="section">
    <div class="eyebrow">Research Context</div>
    <h2>Bridging industry and open research</h2>
    <div class="research-grid">
        <div class="r-cell">
            <div class="r-tag">The Problem</div>
            <div class="r-title">Proprietary and inaccessible</div>
            <div class="r-body">Google has run carbon-aware scheduling internally since 2020. Microsoft released a partial SDK. Neither is available to independent researchers or university labs &mdash; the people responsible for a fast-growing share of AI compute.</div>
        </div>
        <div class="r-cell">
            <div class="r-tag">The Approach</div>
            <div class="r-title">Open, deployable, reproducible</div>
            <div class="r-body">CAML-TC is a lightweight RL-enhanced optimisation layer that runs on top of any existing training loop, using the UK government&#39;s own free public carbon intensity API &mdash; infrastructure built for exactly this purpose.</div>
        </div>
        <div class="r-cell">
            <div class="r-tag">UK Net Zero</div>
            <div class="r-title">Data centres are a growing problem</div>
            <div class="r-body">UK data centres and AI infrastructure are among the fastest-growing electricity consumers. CAML-TC provides the scheduling intelligence layer that was missing &mdash; making carbon-aware ML accessible to anyone.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── FOOTER
st.markdown("""
<div class="site-footer">
    <div class="footer-l">
        <strong>CAML-TC</strong> &middot; Carbon-Aware ML Training Controller &middot;
        Sufiyan Ul Rehman &middot; Ulster University &amp; Solent University (via QA)
    </div>
    <div class="footer-r">
        <a href="https://pypi.org/project/caml-tc" target="_blank">PyPI &#8599;</a>
        <a href="https://github.com/sufirehman/carbon-aware-ml-scheduler" target="_blank">GitHub &#8599;</a>
        <a href="https://ieeexplore.ieee.org" target="_blank">IEEE Paper &#8599;</a>
        <span style="color:#213040;font-family:'JetBrains Mono',monospace;font-size:11px;">MIT License &middot; 2026</span>
    </div>
</div>
""", unsafe_allow_html=True)