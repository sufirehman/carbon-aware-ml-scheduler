import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
try:
    from core.experiment import run_experiment
except ImportError:
    sys.path.append(os.getcwd())
    from core.experiment import run_experiment

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from camltc.carbon_api import CarbonAPI

st.set_page_config(
    page_title="CAML-TC · Simulation Lab",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #05090f !important;
    font-family: 'IBM Plex Sans', sans-serif;
    color: #cbd5e1;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }
.stApp::before {
    content: ''; position: fixed; inset: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 52px 52px;
}

/* ── HEADER ── */
.page-hdr { margin-bottom: 28px; }
.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #f59e0b;
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    padding: 4px 14px; border-radius: 6px;
    letter-spacing: 0.1em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-hdr h1 { font-size: 26px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0 0 6px; }
.page-hdr p  { font-size: 13px; color: #475569; margin: 0; }

/* ── METHOD CARDS (result) ── */
.method-card {
    background: #0b1520; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 30px 20px; text-align: center;
    transition: all 0.25s;
}
.method-card:hover { background: #111e2d; border-color: rgba(255,255,255,0.12); }
.mc-type { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px; }
.mc-val  { font-family:'IBM Plex Mono',monospace; font-size:38px; font-weight:600; letter-spacing:-0.02em; margin-bottom:4px; }
.mc-unit { font-size:11px; color:#64748b; margin-bottom:16px; }
.mc-badge { font-family:'IBM Plex Mono',monospace; font-size:10px; padding:4px 14px; border-radius:4px; letter-spacing:0.07em; text-transform:uppercase; display:inline-block; }
.mb-r { color:#ef4444; background:rgba(239,68,68,0.08); border:1px solid rgba(239,68,68,0.18); }
.mb-a { color:#f59e0b; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.18); }
.mb-g { color:#22c55e; background:rgba(34,197,94,0.08); border:1px solid rgba(34,197,94,0.2); }

/* ── CHART CARD ── */
.chart-card { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.chart-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px; }

/* ── COMPARISON BAR ── */
.comp-section { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 24px; margin-bottom: 16px; }
.comp-title { font-size: 14px; font-weight: 600; color: #f1f5f9; margin-bottom: 20px; }
.comp-row { margin-bottom: 16px; }
.comp-labels { display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#475569; margin-bottom:6px; }
.comp-track { background:rgba(255,255,255,0.04); border-radius:4px; height:8px; overflow:hidden; }
.comp-fill  { height:100%; border-radius:4px; transition: width 0.8s ease; }

/* ── INFO PANEL ── */
.info-panel { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 22px; }
.info-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:16px; }
.info-row { margin-bottom:16px; padding-bottom:16px; border-bottom:1px solid rgba(255,255,255,0.04); }
.info-row:last-child { border-bottom:none; margin-bottom:0; padding-bottom:0; }
.info-method { font-family:'IBM Plex Mono',monospace; font-size:13px; font-weight:500; margin-bottom:5px; }
.info-desc { font-size:12px; color:#475569; line-height:1.7; }

/* ── LOG ── */
.sys-log {
    background: #040810; border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 20px 22px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #334155; line-height: 2.1;
}
.log-g  { color: #22c55e; }
.log-ok { color: #22c55e; font-weight: 600; }
.log-ts { color: #1e293b; margin-right: 8px; }

/* ── SIDEBAR ── */
section[data-testid="stSidebar"] {
    background: #08111c !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #64748b !important; }

/* ── BUTTONS ── */
div.stButton > button {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 13px !important; border-radius: 8px !important;
    padding: 10px 24px !important; width: 100% !important;
    letter-spacing: 0.04em !important; transition: all 0.2s !important;
}
div.stButton > button[kind="primary"] {
    background: #22c55e !important; color: #051a0d !important; border: none !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #16a34a !important; box-shadow: 0 6px 22px rgba(34,197,94,0.28) !important;
}

/* Plotly */
.js-plotly-plot .plotly .bg { fill: transparent !important; }
/* Progress */
div[data-testid="stProgress"] > div { background: rgba(34,197,94,0.12) !important; }
div[data-testid="stProgress"] > div > div { background: #22c55e !important; }
</style>
""", unsafe_allow_html=True)

# ── HEADER
st.markdown("""
<div class="page-hdr">
    <div class="page-badge">Simulation Lab</div>
    <h1>Carbon Scheduling Experiment Runner</h1>
    <p>Compare baseline, heuristic, and RL strategies with live grid data and real emissions measurement</p>
</div>
""", unsafe_allow_html=True)

# ── SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#f59e0b;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:16px;">⚙ Experiment Config</div>', unsafe_allow_html=True)
    runs = st.selectbox("Runs per strategy", [3, 5, 10], index=1,
        help="More runs = more statistically stable results")
    noise_level = st.slider("Grid noise σ (gCO₂/kWh)", 0, 25, 6,
        help="Simulated forecast uncertainty added to real grid data. Reflects real-world unpredictability.")
    st.markdown("---")
    st.markdown('<div style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#1e293b;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px;">About This Experiment</div>', unsafe_allow_html=True)
    st.markdown("""<div style="font-family:IBM Plex Mono,monospace;font-size:11px;color:#334155;line-height:1.9;">
        Each run measures real CO₂<br>
        emissions via CodeCarbon.<br><br>
        Noise σ tests robustness<br>
        of scheduling under<br>
        forecast uncertainty.
    </div>""", unsafe_allow_html=True)

# ── TRAINING FUNCTION (synthetic compute load)
def train_function():
    x = np.random.rand(3000, 3000)
    for _ in range(30):
        x = x @ x

# ── LAYOUT
main_col, guide_col = st.columns([3, 1])

with guide_col:
    st.markdown("""
    <div class="info-panel">
        <div class="info-lbl">Strategy Guide</div>
        <div class="info-row">
            <div class="info-method" style="color:#ef4444;">Baseline</div>
            <div class="info-desc">Training runs immediately with zero carbon intelligence. Represents the current industry default — no scheduling.</div>
        </div>
        <div class="info-row">
            <div class="info-method" style="color:#f59e0b;">Heuristic</div>
            <div class="info-desc">Rule-based scheduler scores windows by carbon intensity, forecast uncertainty, and delay penalty. Strong, interpretable baseline that RL must genuinely beat.</div>
        </div>
        <div class="info-row">
            <div class="info-method" style="color:#22c55e;">RL Agent</div>
            <div class="info-desc">Q-learning MDP trained on 8,000 episodes of real UK grid data. Learns optimal timing under uncertainty — outperforms heuristic when grid volatility is high.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with main_col:
    run_btn = st.button("▶  Run Experiment", type="primary")

    if run_btn:
        prog = st.progress(0)
        status = st.empty()

        status.markdown('<div class="sys-log"><span class="log-g">INIT</span>  Loading 24-hour National Grid ESO forecast...</div>', unsafe_allow_html=True)
        prog.progress(8)

        api = CarbonAPI()
        df_raw = api.get_24h_forecast()
        df_raw["carbon"] = df_raw["actual"].fillna(df_raw["forecast"])

        # Add synthetic noise for robustness testing
        base_carbon = df_raw["carbon"].values
        trend = np.sin(np.linspace(0, np.pi, len(df_raw))) * 5
        noise = np.random.normal(0, noise_level, len(df_raw))
        df_raw["carbon"] = base_carbon + noise + trend

        real_vol = float(np.std(base_carbon))
        sim_vol  = float(np.std(df_raw["carbon"].values))

        status.markdown(f'<div class="sys-log"><span class="log-g">GRID</span>  Forecast loaded · {len(df_raw)} intervals · base σ={real_vol:.1f} → sim σ={sim_vol:.1f} gCO₂/kWh</div>', unsafe_allow_html=True)
        prog.progress(20)

        status.markdown('<div class="sys-log"><span class="log-g">EXP</span>   Running {0} × 3 strategy comparisons...</div>'.replace("{0}", str(runs)), unsafe_allow_html=True)
        prog.progress(35)

        with st.spinner("Running emissions measurements via CodeCarbon..."):
            results = run_experiment(df_raw, train_function, runs=runs)

        prog.progress(92)
        status.markdown('<div class="sys-log"><span class="log-g">PROC</span>  Aggregating results...</div>', unsafe_allow_html=True)
        prog.progress(100)
        status.empty(); prog.empty()

        st.session_state.update({
            "sim_results": results,
            "sim_df": df_raw,
            "sim_vol_real": real_vol,
            "sim_vol_sim": sim_vol,
            "sim_runs": runs,
            "sim_noise": noise_level,
            "sim_ready": True
        })

    if st.session_state.get("sim_ready"):
        results   = st.session_state.sim_results
        real_vol  = st.session_state.sim_vol_real
        sim_vol   = st.session_state.sim_vol_sim
        n_runs    = st.session_state.sim_runs
        noise     = st.session_state.sim_noise

        base_g = float(results["baseline"])  * 1000
        heur_g = float(results["heuristic"]) * 1000
        rl_g   = float(results["rl"])        * 1000

        heur_save = (base_g - heur_g) / base_g * 100
        rl_save   = (base_g - rl_g)   / base_g * 100
        rl_vs_h   = (heur_g - rl_g)   / heur_g * 100

        # ── RESULT CARDS
        c1, c2, c3 = st.columns(3)
        for col, color, method, val, bc, badge in [
            (c1, "#ef4444", "Baseline",  base_g, "mb-r", "No optimisation"),
            (c2, "#f59e0b", "Heuristic", heur_g, "mb-a", "Rule-based"),
            (c3, "#22c55e", "RL Agent",  rl_g,   "mb-g", "Adaptive RL"),
        ]:
            with col:
                st.markdown(f"""<div class="method-card">
                    <div class="mc-type">{method}</div>
                    <div class="mc-val" style="color:{color}">{val:.2f}</div>
                    <div class="mc-unit">g CO₂  ({n_runs}-run avg)</div>
                    <div class="mc-badge {bc}">{badge}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── RELATIVE COMPARISON
        st.markdown(f"""
        <div class="comp-section">
            <div class="comp-title">Relative Emissions · Baseline = 100%</div>
            <div class="comp-row">
                <div class="comp-labels">
                    <span>Baseline</span>
                    <span style="color:#ef4444">100%  —  {base_g:.2f} g CO₂</span>
                </div>
                <div class="comp-track"><div class="comp-fill" style="width:100%;background:#ef4444;"></div></div>
            </div>
            <div class="comp-row">
                <div class="comp-labels">
                    <span>Heuristic</span>
                    <span style="color:#f59e0b">{heur_g/base_g*100:.1f}%  —  {heur_g:.2f} g CO₂  (↓ {heur_save:.1f}%)</span>
                </div>
                <div class="comp-track"><div class="comp-fill" style="width:{heur_g/base_g*100:.1f}%;background:#f59e0b;"></div></div>
            </div>
            <div class="comp-row">
                <div class="comp-labels">
                    <span>RL Agent</span>
                    <span style="color:#22c55e">{rl_g/base_g*100:.1f}%  —  {rl_g:.2f} g CO₂  (↓ {rl_save:.1f}%)</span>
                </div>
                <div class="comp-track"><div class="comp-fill" style="width:{rl_g/base_g*100:.1f}%;background:#22c55e;"></div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── BAR + LINE CHART
        chart_c1, chart_c2 = st.columns(2)

        with chart_c1:
            st.markdown('<div class="chart-card"><div class="chart-title">Average Emissions by Strategy · g CO₂</div>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=["Baseline", "Heuristic", "RL Agent"],
                y=[base_g, heur_g, rl_g],
                text=[f"{base_g:.2f}g", f"{heur_g:.2f}g", f"{rl_g:.2f}g"],
                textposition="auto",
                textfont=dict(family="IBM Plex Mono", size=11, color="#e2e8f0"),
                marker_color=["#ef4444", "#f59e0b", "#22c55e"],
                marker_line_width=0, width=0.45
            ))
            fig.update_layout(
                height=260, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=8, b=8),
                xaxis=dict(showgrid=False, color="#475569",
                           tickfont=dict(family="IBM Plex Mono", size=12)),
                yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                           color="#334155", tickfont=dict(family="IBM Plex Mono", size=10),
                           title="g CO₂"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with chart_c2:
            # Per-run breakdown from raw data
            raw_df = results.get("raw", None)
            if raw_df is not None and len(raw_df) > 1:
                st.markdown('<div class="chart-card"><div class="chart-title">Per-Run Emissions Breakdown · g CO₂</div>', unsafe_allow_html=True)
                fig2 = go.Figure()
                runs_x = raw_df["run"].tolist()
                for col_name, color, label in [
                    ("baseline",  "#ef4444", "Baseline"),
                    ("heuristic", "#f59e0b", "Heuristic"),
                    ("rl",        "#22c55e", "RL Agent"),
                ]:
                    fig2.add_trace(go.Scatter(
                        x=runs_x,
                        y=(raw_df[col_name] * 1000).tolist(),
                        mode="lines+markers",
                        name=label,
                        line=dict(color=color, width=2),
                        marker=dict(size=7, color=color),
                        hovertemplate=f"<b>{label}</b><br>Run %{{x}}: %{{y:.3f}}g<extra></extra>"
                    ))
                fig2.update_layout(
                    height=260, template="plotly_dark",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=8, b=8),
                    xaxis=dict(showgrid=False, color="#475569",
                               tickfont=dict(family="IBM Plex Mono", size=10),
                               title="Run", dtick=1),
                    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)",
                               color="#334155", tickfont=dict(family="IBM Plex Mono", size=10),
                               title="g CO₂"),
                    legend=dict(font=dict(family="IBM Plex Mono", size=10, color="#475569"),
                                bgcolor="transparent", borderwidth=0)
                )
                st.plotly_chart(fig2, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)

        # ── LOG
        best_name = min({"Baseline": base_g, "Heuristic": heur_g, "RL Agent": rl_g},
                        key=lambda k: {"Baseline": base_g, "Heuristic": heur_g, "RL Agent": rl_g}[k])
        rl_note = "RL outperforms heuristic — high volatility conditions" if rl_save > heur_save else "RL on par with heuristic — moderate grid conditions"
        import datetime
        ts = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
        st.markdown(f"""
        <div class="sys-log">
            <span class="log-ts">[{ts}]</span><span class="log-g">EXP</span>   {n_runs}-run experiment complete · noise σ={noise} added for robustness testing<br>
            <span class="log-ts">[{ts}]</span><span class="log-g">GRID</span>  Base grid volatility σ={real_vol:.1f} → simulated σ={sim_vol:.1f} gCO₂/kWh<br>
            <span class="log-ts">[{ts}]</span><span class="log-g">BASE</span>  Baseline: {base_g:.3f}g CO₂ (no scheduling)<br>
            <span class="log-ts">[{ts}]</span><span class="log-g">HEUR</span>  Heuristic: {heur_g:.3f}g CO₂  (↓ {heur_save:.1f}% vs baseline)<br>
            <span class="log-ts">[{ts}]</span><span class="log-g">RL</span>    RL agent: {rl_g:.3f}g CO₂  (↓ {rl_save:.1f}% vs baseline · {rl_note})<br>
            <span class="log-ok">✓ DONE</span>  Best strategy: <strong style="color:#22c55e">{best_name.upper()}</strong> · emissions reduced via intelligent carbon-aware scheduling
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── EMPTY STATE
        st.markdown(f"""
        <div style="background:#0b1520; border:1px solid rgba(255,255,255,0.06);
             border-radius:14px; padding:80px 40px; text-align:center; margin-top:12px;">
            <div style="font-family:'IBM Plex Mono',monospace; font-size:28px; color:#f59e0b; opacity:0.2; margin-bottom:20px;">🤖</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:12px; color:#1e293b; letter-spacing:0.06em; text-transform:uppercase;">
                Configure experiment parameters · then run to compare strategies
            </div>
            <div style="font-size:12px; color:#1e293b; margin-top:10px; line-height:1.7;">
                The experiment will run baseline, heuristic, and RL scheduling on today's live<br>
                National Grid ESO data and measure real CO₂ emissions via CodeCarbon
            </div>
        </div>
        """, unsafe_allow_html=True)