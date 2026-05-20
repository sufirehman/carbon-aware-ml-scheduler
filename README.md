# 🌍 Carbon-Aware ML Training Controller — CAML-TC

[![PyPI version](https://badge.fury.io/py/caml-tc.svg)](https://badge.fury.io/py/caml-tc)
![Downloads](https://static.pepy.tech/badge/caml-tc)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://pypi.org/project/caml-tc)
[![Streamlit](https://img.shields.io/badge/Live%20App-Streamlit-red?logo=streamlit)](https://carbon-aware-ml-scheduler-ilmd5zrsv44sc6ub3vhbtu.streamlit.app/)
[![IEEE EEEIC 2026](https://img.shields.io/badge/IEEE%20EEEIC-2026%20%7C%20Q1-blue)](https://ieeexplore.ieee.org)
[![Green AI](https://img.shields.io/badge/AI-Carbon%20Aware-success)](https://github.com/sufirehman/carbon-aware-ml-scheduler)
[![Status](https://img.shields.io/badge/Status-Active%20Research-brightgreen)](https://github.com/sufirehman/carbon-aware-ml-scheduler)

---

## ⚡ Executive Summary

A **reinforcement learning–driven carbon-aware scheduling system** that dynamically optimises machine learning workloads using real-time and forecasted UK electricity grid carbon intensity.

> 📄 **Peer-reviewed research** — Accepted at **IEEE EEEIC 2026** (Q1, Scopus-indexed, Web of Science)  
> 🔗 **[Try the live app](https://carbon-aware-ml-scheduler-ilmd5zrsv44sc6ub3vhbtu.streamlit.app/)** — uses live UK National Grid data, no login required  
> 📦 **`pip install caml-tc`** — use as a Python library in your own ML pipeline

The system integrates:

- 🔮 **Carbon forecasting** with uncertainty-aware confidence weighting
- 🧠 **Heuristic optimisation** with multi-window holistic scoring
- 🤖 **Risk-aware reinforcement learning** — Q-learning MDP trained on 8,000 episodes of real UK grid data

to shift compute workloads into low-emission periods, achieving **28–34% CO₂ reduction under realistic conditions** and **>50% under optimal conditions** — without changing a single line of model code.

---

## 💡 Problem Statement

The UK electricity grid fluctuates between **50 and 260 gCO₂/kWh** within a single day. The same ML training job at 3am versus 7pm evening peak can produce **3–5× different carbon emissions**.

Modern machine learning workloads are executed without any environmental awareness. This results in:

- ❌ Unnecessary CO₂ emissions from poorly timed execution
- ❌ No integration between carbon intelligence and scheduling decisions
- ❌ Lack of sustainability in ML pipelines despite available grid data
- ❌ Existing solutions (Google, Meta, Microsoft) are proprietary and inaccessible

**Google has done this internally since 2020. Microsoft released a partial SDK. Nobody built an open, deployable version for individual researchers and engineers. Until now.**

---

## 📊 Validated Results

Tested on real UK National Grid data. Emissions measured with **CodeCarbon**.

| Strategy | Carbon Reduction | Conditions |
|---|---|---|
| Baseline (immediate execution) | 0% | — |
| Heuristic Scheduler | ~28% | Realistic UK grid |
| **RL Scheduler (CAML-TC)** | **28–34%** | Realistic UK grid |
| **RL Scheduler (CAML-TC)** | **>50%** | Optimal low-variability |

> RL outperforms heuristics under **high grid variability** — the conditions where static rules fail and adaptive learning matters most.

---

## 🚀 Install & Quickstart

```bash
pip install caml-tc
```

```python
from camltc import CarbonScheduler

scheduler = CarbonScheduler(duration_minutes=90, urgency="low")
result = scheduler.recommend()

print(result.best_window)        # "03:00 — 04:30 UTC"
print(result.carbon_saving_pct)  # 31.4
print(result.strategy)           # "rl" or "heuristic"
print(result)                    # full formatted summary
```

### Urgency levels

```python
# Low — aggressive carbon optimisation, longer delay allowed
CarbonScheduler(duration_minutes=120, urgency="low").recommend()

# Medium — balanced (default)
CarbonScheduler(duration_minutes=60, urgency="medium").recommend()

# High — run as soon as a clean-enough window appears
CarbonScheduler(duration_minutes=30, urgency="high").recommend()
```

---

## 🧠 Key Innovations

- 🌱 Carbon intensity used as a **first-class scheduling signal** — not an afterthought
- 🤖 **Hybrid scheduling architecture** — heuristic for stable grids, RL for volatile ones
- ⚖️ **Multi-objective RL reward design** — jointly optimises carbon intensity, forecast uncertainty, delay penalty, and deadline constraints
- 📉 **Exponential forecast confidence decay** — `wₜ = exp(−λ·Δt)` — prevents over-commitment to unreliable long-horizon predictions
- 🎲 **Stochastic noise injection** — forces RL agent to learn policies robust to real grid variability, not just noise-free signals
- 📡 **End-to-end closed-loop pipeline** — from raw National Grid API to scheduling decision in one system
- 🧪 **CodeCarbon-integrated emissions measurement** — validated, reproducible results

---

## ⚙️ System Architecture

```
┌──────────────────────────────────────┐
│   UK National Grid Carbon API        │
│   Real-time + 24h Forecast           │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│   Carbon Intelligence Layer          │
│   Confidence decay · Volatility      │
│   Peak/low detection · Uncertainty   │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│   Scheduling Engine                  │
│   ├─ Heuristic Optimizer             │
│   │    Window scoring · Delay penalty│
│   └─ RL Policy Agent (Q-Learning MDP)│
│        State: (μ, σ, delay)          │
│        Reward: carbon+uncertainty    │
│               +delay+deadline        │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│   ML Workload Simulator              │
│   Synthetic training load            │
│   CodeCarbon emissions tracking      │
└──────────────────┬───────────────────┘
                   ↓
┌──────────────────────────────────────┐
│   Analytics Dashboard (Streamlit)    │
│   Forecast Explorer · Simulation Lab │
│   Strategy Comparison · KPI View     │
└──────────────────────────────────────┘
```

---

## ✨ Key Features

- 📡 Real-time UK carbon intensity integration via National Grid API
- 🔮 24-hour carbon forecasting with uncertainty modelling
- 🧠 Heuristic + RL scheduling engine — automatically selects best strategy
- 🤖 Risk-aware RL agent trained on 8,000 episodes of real UK grid data
- 🧪 ML workload simulation with CodeCarbon emissions tracking
- 📊 Interactive multi-page Streamlit dashboard
- 📈 Carbon peak/low detection and optimal window identification
- ⚙️ Multi-strategy emissions comparison (baseline vs heuristic vs RL)
- 📦 pip-installable Python library for direct pipeline integration

---

## 🧪 System Modules

### 📈 `core/carbon_api.py` — Carbon Intelligence Layer
- Real-time and forecast carbon intensity from UK National Grid
- Exponential confidence weighting on future predictions
- Peak/low detection and optimal window identification

### ⚙️ `core/scheduler.py` — Heuristic Scheduling Engine
- Holistic window scoring (carbon intensity + uncertainty + delay penalty)
- Urgency-parameterised delay penalty coefficient
- Strong interpretable baseline that RL must genuinely beat

### 🤖 `core/rl_agent.py` — Reinforcement Learning Agent
- Q-learning with ε-greedy exploration and linear epsilon decay
- State: `(μₜ, σₜ, dₜ)` — mean CI, uncertainty proxy, accumulated delay
- Multi-objective reward: carbon + uncertainty + delay + deadline violation
- 8,000 training episodes · 4-phase iterative development

### 🧪 `core/simulator.py` — Emissions Measurement
- Synthetic ML workload generation
- CodeCarbon-based emissions tracking per strategy
- Reproducible experimental pipeline

### 📊 `demo/` — Interactive Dashboard
- `app.py` — multi-page Streamlit application
- `pages/forecast.py` — 24h carbon intensity explorer
- `pages/simulation.py` — strategy comparison lab
- `pages/overview.py` — KPI and results summary

---

## 🧰 Tech Stack

- **Python 3.9+**
- **Streamlit** — interactive dashboard
- **Plotly** — visualisations
- **Pandas / NumPy** — data processing
- **CodeCarbon** — emissions measurement
- **Custom RL Agent** — Q-learning MDP
- **UK Carbon Intensity API** — `api.carbonintensity.org.uk`

---

## 🌍 Environmental Impact

| CO₂ Saved | Real-World Equivalent |
|---|---|
| 1 kg | 🌳 ~0.045 trees planted |
| 10 kg | 🚗 ~40 km of driving avoided |
| 100 kg | ✈️ One short-haul flight offset |

> **Example:** A team running 10 GPU training jobs per week, each saving 28–34% emissions, saves hundreds of kg of CO₂ annually — with zero infrastructure changes.

---

## 🇬🇧 UK Net Zero Relevance

This system is **directly aligned with the UK's Net Zero 2050 commitment**. It uses the National Grid's own free public carbon intensity API — infrastructure the UK government has already built.

Data centres and AI infrastructure are among the fastest-growing electricity consumers in the UK. CAML-TC provides the scheduling intelligence layer that was missing — making carbon-aware ML accessible to any researcher or engineer, not just those inside Google or Microsoft.

---

## 🚀 Roadmap

- [ ] PyTorch training callback (`on_epoch_end` auto-scheduling)
- [ ] HuggingFace Trainer integration
- [ ] AWS / Azure / GCP cloud scheduler hooks
- [ ] Transformer-based CI forecasting (replacing LSTM baseline)
- [ ] Continuous-action RL — DQN, PPO
- [ ] Multi-region geographic shifting (EU / US grids)
- [ ] Carbon-aware CI/CD for ML training pipelines
- [ ] Kubernetes-based orchestration

---

## 📄 Research & Citation

**Published at IEEE EEEIC 2026** — International Conference on Environment and Electrical Engineering (Q1, Scopus-indexed, Web of Science)

If you use CAML-TC in your work, please cite:

```bibtex
@inproceedings{rehman2026camltc,
  title     = {Adaptive Carbon-Aware Machine Learning Training under Uncertainty:
               A Unified Scheduling Framework},
  author    = {Rehman, Sufiyan Ul and Fatima, Areej and Shahid, Zohaib and Iqbal, Nasir},
  booktitle = {2026 IEEE International Conference on Environment and
               Electrical Engineering (EEEIC)},
  year      = {2026},
  publisher = {IEEE}
}
```

---

## 👨‍💻 Author

**Sufiyan Ul Rehman**  
AI/ML Researcher · Lecturer, Ulster University & Solent University (via QA Higher Education), London  
Building intelligent, carbon-aware AI systems for sustainable machine learning infrastructure.

🔗 [Live App](https://carbon-aware-ml-scheduler-ilmd5zrsv44sc6ub3vhbtu.streamlit.app/) · 
📦 [PyPI](https://pypi.org/project/caml-tc) · 
🔬 [IEEE Paper](https://ieeexplore.ieee.org)

---

## 📌 Closing Statement

The future of machine learning is not only about **accuracy and performance**,  
but also about **when and how sustainably computation is executed**.

The carbon cost of AI is real. The fix is simpler than anyone realises.  
**It is just a matter of timing.**

---

## 📜 License

MIT © 2026 Sufiyan Ul Rehman — see [LICENSE](LICENSE)
