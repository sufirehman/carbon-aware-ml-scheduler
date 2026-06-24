# CAML-TC Design System — MASTER

**Project:** Carbon-Aware ML Training Scheduler  
**Category:** Predictive Analytics / Real-Time Monitoring Dashboard  
**Stack:** Streamlit + Plotly + custom CSS (injected via `st.markdown`)  
**Archetype:** Engineering research tool — think Grafana, Datadog, academic paper dashboard. Not a SaaS marketing page.

---

## 1. Governing Principles

These override any auto-generated recommendation that conflicts with them.

| Rule | Rationale |
|------|-----------|
| **No fake real-time gimmicks** | Scrolling tickers, pulsing glows, and "LIVE" badges that display static data erode credibility. Researchers notice immediately. |
| **No emoji as icons** | Emojis render inconsistently across OS/browser and break the professional register. Use SVG inline or Lucide/Heroicons. |
| **No AI-purple gradient** | `purple → violet → indigo` gradients are the current visual cliché for AI products. This tool is a sustainability/ops tool, not an AI chatbot. |
| **Dark, not gothic** | Dark background is correct for an ops dashboard (reduces eye strain during long sessions). Keep it practical-dark (`#05090f`), not dramatic/neon. |
| **Data density over visual drama** | The tool's value is in the numbers. Typography and layout should direct attention to metrics, not to the chrome around them. |
| **Monospace for data, sans for prose** | IBM Plex Mono for all numeric values, labels, timestamps, and code. IBM Plex Sans for explanatory text. Never reverse this. |

---

## 2. Color Palette

### 2.1 Surface Stack (dark → light)

| Token | Hex | Usage |
|-------|-----|-------|
| `surface-base` | `#05090f` | Page background (`html, body, .stApp`) |
| `surface-card` | `#0b1520` | KPI cards, chart cards, info panels |
| `surface-sidebar` | `#08111c` | Sidebar background |
| `surface-terminal` | `#040810` | System log / terminal output |
| `surface-deep` | `#04070c` | Footer, darkest recessed areas |

### 2.2 Borders

| Token | Value | Usage |
|-------|-------|-------|
| `border-default` | `rgba(255,255,255,0.07)` | Card and panel borders |
| `border-subtle` | `rgba(255,255,255,0.04)` | Row dividers, table cells |
| `border-section` | `rgba(255,255,255,0.06)` | Full-width section dividers |
| `border-sidebar` | `rgba(255,255,255,0.06)` | Sidebar right edge |

### 2.3 Semantic Accent Colors

Each accent maps to a domain concept — do not reassign them.

| Token | Hex | Semantic meaning | Used for |
|-------|-----|-----------------|---------|
| `accent-green` | `#22c55e` | Carbon savings, optimal, success | Best window, CO₂ reduction, RL agent win, "run" actions |
| `accent-green-muted` | `rgba(34,197,94,0.08)` | Green surface tint | Badge backgrounds, card hover tint |
| `accent-green-border` | `rgba(34,197,94,0.22)` | Green border | Badge borders, result hero border |
| `accent-red` | `#ef4444` | High carbon, baseline, danger | Worst window, baseline strategy, peak carbon |
| `accent-amber` | `#f59e0b` | Heuristic strategy, warning, mid | Heuristic results, moderate conditions |
| `accent-sky` | `#0ea5e9` | Forecast / predicted values | Dashed forecast lines, forecast page badge |
| `accent-green-dark` | `#16a34a` | Hover state for green CTA | Button hover |

### 2.4 Text Scale

| Token | Hex | Usage |
|-------|-----|-------|
| `text-primary` | `#f1f5f9` | Page headings, key values |
| `text-secondary` | `#cbd5e1` | Body text |
| `text-muted` | `#94a3b8` | Chart labels, sub-values, secondary prose |
| `text-dim` | `#64748b` | Descriptive captions, hints |
| `text-faint` | `#475569` | Section subtitles, less important labels |
| `text-ghost` | `#334155` | Log timestamps, metadata, background labels |
| `text-invisible` | `#1e293b` | Ultra-dim log noise, sidebar metadata |

### 2.5 Page-Badge Colors (one per page, do not swap)

| Page | Badge color | Rationale |
|------|-------------|-----------|
| `app.py` (landing) | `#22c55e` (green) | Primary brand color |
| `overview.py` (dashboard) | `#22c55e` (green) | Active/run mode |
| `forecast.py` | `#0ea5e9` (sky) | Forecast / prediction domain |
| `simulation.py` | `#f59e0b` (amber) | Experimental / lab mode |

---

## 3. Typography

### 3.1 Typefaces

```css
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&display=swap');
```

| Role | Family | Weight range |
|------|--------|-------------|
| Monospace (data) | `IBM Plex Mono` | 300, 400, 500, 600 |
| Sans (prose) | `IBM Plex Sans` | 300, 400, 500, 600, 700 |

### 3.2 Type Scale

| Element | Font | Size | Weight | Color |
|---------|------|------|--------|-------|
| Page h1 | IBM Plex Sans | 26px | 700 | `#f1f5f9` |
| Section h2 | IBM Plex Sans | 34px | 600 | `#f1f5f9` |
| KPI large value | IBM Plex Mono | 30–42px | 600 | `#f1f5f9` or accent |
| Chart title | IBM Plex Mono | 10px | 400 | `#475569` |
| Badge / eyebrow | IBM Plex Mono | 10–11px | 400 | accent color |
| Body prose | IBM Plex Sans | 13–14px | 400 | `#64748b` |
| Caption | IBM Plex Sans | 11–12px | 400 | `#475569` |
| System log | IBM Plex Mono | 12px | 400 | `#334155` |
| Nav links | IBM Plex Mono | 12px | 400 | `#94a3b8` |
| Code inline | IBM Plex Mono | 12px | 400 | inherit |

Letter spacing rules:
- All-caps labels: `0.08–0.12em`
- Nav / badge text: `0.05–0.06em`
- Large KPI numbers: `-0.02em` (tighten)
- Page h1: `-0.02em`

---

## 4. Background Texture

All pages include a very-low-opacity green grid overlay:

```css
.stApp::before {
    content: '';
    position: fixed; inset: 0; z-index: 0; pointer-events: none;
    background-image:
        linear-gradient(rgba(34,197,94,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(34,197,94,0.025) 1px, transparent 1px);
    background-size: 52px 52px;
}
```

**Rule:** Opacity must not exceed `0.03`. At higher values it competes with chart gridlines and adds visual noise. This is a texture, not a feature.

---

## 5. Component Library

### 5.1 Page Header

Appears at the top of every dashboard page (not `app.py`). Provides orientation.

```html
<div class="page-hdr">
    <div class="page-badge">PAGE NAME</div>
    <h1>Descriptive Title</h1>
    <p>One-line description · data source · resolution</p>
</div>
```

```css
.page-hdr { margin-bottom: 28px; }
.page-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px;
    background: rgba(<badge-rgb>, 0.08);
    border: 1px solid rgba(<badge-rgb>, 0.2);
    padding: 4px 14px; border-radius: 6px;
    letter-spacing: 0.1em; text-transform: uppercase;
    display: inline-block; margin-bottom: 10px;
}
.page-hdr h1 { font-size: 26px; font-weight: 700; color: #f1f5f9; letter-spacing: -0.02em; margin: 0 0 6px; }
.page-hdr p  { font-size: 13px; color: #475569; margin: 0; }
```

### 5.2 KPI Cards — Two Variants

**Variant A — Bottom accent bar** (used on `overview.py`):
```css
.kpi {
    background: #0b1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 20px;
    position: relative; overflow: hidden;
}
.kpi::after {
    content: ''; position: absolute;
    bottom: 0; left: 0; right: 0; height: 2px;
}
.kpi-g::after { background: #22c55e; }
.kpi-b::after { background: #0ea5e9; }
.kpi-r::after { background: #ef4444; }
.kpi-a::after { background: #f59e0b; }
.kpi-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px; }
.kpi-val { font-family:'IBM Plex Mono',monospace; font-size:30px; font-weight:600; color:#f1f5f9; letter-spacing:-0.02em; margin-bottom:4px; }
.kpi-sub { font-size:11px; color:#475569; }
```

**Variant B — Left accent border** (used on `forecast.py`):
```css
.stat {
    background: #0b1520;
    border: 1px solid rgba(255,255,255,0.07);
    border-left: 2px solid transparent;
    border-radius: 10px; padding: 18px 20px;
}
.stat-g { border-left-color: #22c55e !important; }
.stat-b { border-left-color: #0ea5e9 !important; }
.stat-r { border-left-color: #ef4444 !important; }
.stat-a { border-left-color: #f59e0b !important; }
.stat-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#334155; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px; }
.stat-val { font-family:'IBM Plex Mono',monospace; font-size:28px; font-weight:600; color:#f1f5f9; letter-spacing:-0.02em; margin-bottom:4px; }
.stat-sub { font-size:11px; color:#475569; }
```

**Usage rule:** Use Variant A (bottom bar) on action/result pages where something has just been computed. Use Variant B (left border) on analytical pages showing ambient data. Do not mix both variants on the same page.

### 5.3 Chart Card

Wrapper for all Plotly charts.

```css
.chart-card { background: #0b1520; border: 1px solid rgba(255,255,255,0.07); border-radius: 12px; padding: 24px; }
.chart-hdr  { display:flex; align-items:center; justify-content:space-between; margin-bottom:16px; }
.chart-title { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#475569; text-transform:uppercase; letter-spacing:0.1em; display:flex; align-items:center; gap:8px; }
.chart-sub  { font-size:12px; color:#94a3b8; margin-bottom:14px; }
.chart-meta { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#1e293b; }
.cdot       { width:7px; height:7px; border-radius:50%; display:inline-block; }
```

**Plotly defaults for all charts** (apply to every `fig.update_layout()`):
```python
fig.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    xaxis=dict(showgrid=False, color="#334155",
               tickfont=dict(family="IBM Plex Mono", size=10)),
    yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.04)", color="#334155",
               tickfont=dict(family="IBM Plex Mono", size=10)),
    hovermode="x unified",
    margin=dict(l=0, r=10, t=10, b=10),
)
```

**Chart color assignment** (consistent across all pages):
- Actual/observed data: `#22c55e` solid line
- Forecast/predicted: `#0ea5e9` dashed line (`dash="dot"`)
- Baseline/worst: `#ef4444`
- Background fill: Use `fillcolor="rgba(71,85,105,0.04)"` for area charts
- Optimal region highlight: `fillcolor="rgba(34,197,94,0.12)"`, `line_color="rgba(34,197,94,0.3)"`
- Danger region highlight: `fillcolor="rgba(239,68,68,0.07)"`, `line_color="rgba(239,68,68,0.2)"`

### 5.4 System Log

Terminal-style output for process reporting. Used in `overview.py` and `simulation.py`.

```css
.sys-log {
    background: #040810;
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 10px; padding: 20px 22px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px; color: #334155; line-height: 2.1;
}
.log-g  { color: #22c55e; }
.log-ok { color: #22c55e; font-weight: 600; }
.log-ts { color: #1e293b; margin-right: 8px; }
```

**Log line format:** `[HH:MM:SS UTC] TAG   Message · detail`  
Tags are 4–5 chars uppercase: `INIT`, `GRID`, `SCHED`, `OPT`, `BASE`, `EXP`, `HEUR`, `RL`, `DONE`

### 5.5 Strategy / Method Badges

Inline badges indicating which algorithm/result tier.

```css
.strategy-badge { display:inline-block; font-size:10px; letter-spacing:0.1em; text-transform:uppercase; padding:4px 12px; border-radius:4px; }
.strategy-rl    { color:#22c55e; background:rgba(34,197,94,0.08);  border:1px solid rgba(34,197,94,0.2);  }
.strategy-heu   { color:#f59e0b; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); }
.strategy-base  { color:#ef4444; background:rgba(239,68,68,0.08);  border:1px solid rgba(239,68,68,0.18); }
```

### 5.6 Window / Scheduling List

Ranked list of carbon scheduling windows. Used in `forecast.py`.

```css
.win-item { display:flex; align-items:center; justify-content:space-between; padding:14px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.win-item:last-child { border-bottom:none; }
.win-time { font-family:'IBM Plex Mono',monospace; font-size:14px; font-weight:500; }
.win-avg  { font-size:11px; color:#475569; margin-top:3px; }
.win-badge { font-family:'IBM Plex Mono',monospace; font-size:10px; padding:4px 12px; border-radius:4px; letter-spacing:0.08em; text-transform:uppercase; }
.b-best { color:#22c55e; background:rgba(34,197,94,0.08);  border:1px solid rgba(34,197,94,0.2);  }
.b-good { color:#f59e0b; background:rgba(245,158,11,0.08); border:1px solid rgba(245,158,11,0.2); }
.b-ok   { color:#475569; background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); }
```

### 5.7 Comparison Bar

Used in `simulation.py` to show relative emissions as percentage bars.

```css
.comp-section { background:#0b1520; border:1px solid rgba(255,255,255,0.07); border-radius:12px; padding:24px; margin-bottom:16px; }
.comp-title   { font-size:14px; font-weight:600; color:#f1f5f9; margin-bottom:20px; }
.comp-row     { margin-bottom:16px; }
.comp-labels  { display:flex; justify-content:space-between; font-family:'IBM Plex Mono',monospace; font-size:11px; color:#475569; margin-bottom:6px; }
.comp-track   { background:rgba(255,255,255,0.04); border-radius:4px; height:8px; overflow:hidden; }
.comp-fill    { height:100%; border-radius:4px; }
```

### 5.8 Insight / Research Notes Panel

Narrative text block inside a card. Used for context, methodology, and analysis.

```css
.insight {
    background: rgba(34,197,94,0.03);
    border: 1px solid rgba(34,197,94,0.1);
    border-radius: 8px; padding: 20px;
    font-size: 13px; color: #64748b; line-height: 1.8;
}
.ins-lbl { font-family:'IBM Plex Mono',monospace; font-size:10px; color:#22c55e; text-transform:uppercase; letter-spacing:0.12em; margin-bottom:10px; }
```

### 5.9 Buttons

```css
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
    background: #16a34a !important;
    box-shadow: 0 6px 22px rgba(34,197,94,0.28) !important;
}
div.stButton > button[kind="secondary"] {
    background: transparent !important; color: #475569 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
}
div.stButton > button[kind="secondary"]:hover {
    color: #e2e8f0 !important; border-color: rgba(255,255,255,0.2) !important;
}
```

### 5.10 Sidebar

```css
section[data-testid="stSidebar"] {
    background: #08111c !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
section[data-testid="stSidebar"] * { color: #64748b !important; }
.sb-title {
    font-family:'IBM Plex Mono',monospace; font-size:11px;
    color:#22c55e !important; letter-spacing:0.1em;
    text-transform:uppercase; margin-bottom:16px;
}
```

### 5.11 Tabs

```css
.stTabs [data-baseweb="tab-list"] { background:#0b1520 !important; border-bottom:1px solid rgba(255,255,255,0.06) !important; gap:4px; }
.stTabs [data-baseweb="tab"] { font-family:'IBM Plex Mono',monospace !important; font-size:11px !important; color:#334155 !important; letter-spacing:0.07em !important; text-transform:uppercase !important; background:transparent !important; }
.stTabs [aria-selected="true"] { color:#22c55e !important; border-bottom:2px solid #22c55e !important; }
.stTabs [data-baseweb="tab-panel"] { background:#0b1520 !important; border:1px solid rgba(255,255,255,0.06) !important; border-top:none !important; border-radius:0 0 10px 10px !important; padding:20px !important; }
```

---

## 6. Page-Level Structure

### 6.1 `app.py` — Landing / About

**Role:** Research paper abstract + tool introduction for first-time visitors.  
**Tone:** Academic credibility, not marketing hype.

**Layout order:**
1. Navigation bar (sticky, minimal — logo + page links + PyPI badge)
2. Research context block (title, paper citation, one-paragraph description)
3. Key results grid (4 cells: CO₂ reduction %, RL episodes, forecast horizon, deployment model)
4. System architecture (4-step pipeline — use SVG/text icons, not emoji)
5. Key innovations (3-column feature grid — text only, no decorative icons)
6. Benchmarking table (strategy × conditions × reduction %)
7. Research context (3-column card: Problem / Approach / UK alignment)
8. Footer (attribution, links)

**What NOT to include on `app.py`:**
- Scrolling ticker bar with fake live data values
- Any animated pulsing indicator claiming to be "LIVE"
- Emoji icons in the pipeline or feature cards
- 62px hero headline — use 38–44px max for a research-tool landing page
- Hero section styled like a SaaS product launch (gradient headline, big CTA above the fold)

**Navigation structure:**
```html
<div class="top-nav">
    <div class="nav-brand">CAML-TC</div>
    <div class="nav-links">
        <a href="/overview">Dashboard</a>
        <a href="/forecast">Forecast</a>
        <a href="/simulation">Simulation</a>
        <a href="https://pypi.org/project/caml-tc" target="_blank">PyPI ↗</a>
    </div>
    <div class="nav-pill">UK National Grid · ESO API</div>
</div>
```
The nav-pill should describe the data source, not claim "LIVE" status.

**Section eyebrow pattern** (use in place of large decorative sections):
```css
.section-eye {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; color: #22c55e;
    text-transform: uppercase; letter-spacing: 0.14em;
    display: flex; align-items: center; gap: 10px;
}
.section-eye::before { content:''; display:inline-block; width:24px; height:1px; background:#22c55e; }
```

### 6.2 `overview.py` — Carbon Intelligence Dashboard

**Role:** Primary tool — researcher runs a job, sees the optimal window and savings.  
**Layout order:**
1. Page header (badge: Dashboard)
2. Sidebar: duration slider, urgency select, data source info
3. Run button (primary, full-width of left column)
4. [After run] Result hero (large CO₂ saving % + optimal start time + strategy badge)
5. KPI row × 4 (CO₂ saved, optimal emissions, baseline emissions, delay required) — Variant A cards
6. Charts row (24-hr decision timeline 2/3 width + emissions comparison bar 1/3)
7. System log (terminal output)
8. PDF export button

**Empty state:** Centered panel with muted text instructing the user to configure and run. No emoji placeholder.

### 6.3 `forecast.py` — Carbon Intensity Forecast System

**Role:** Analytical view of today's grid — understanding the carbon landscape.  
**Layout order:**
1. Page header (badge: Forecast, badge color: `#0ea5e9`)
2. Stat cards × 4 (peak, low, 24-hr avg, volatility) — Variant B left-border cards, auto-loaded
3. Main chart: 24-hr carbon curve (actual=green solid, forecast=sky dashed, optimal window highlighted)
4. Two-column lower panel:
   - Left: Optimal scheduling windows list
   - Right: Grid intelligence analysis (insight panel)
5. Tabs: Summary (JSON), Raw Forecast Data (table), Research Notes

### 6.4 `simulation.py` — Carbon Scheduling Experiment Runner

**Role:** Lab/research mode — run multi-strategy experiment and compare results.  
**Layout order:**
1. Page header (badge: Simulation Lab, badge color: `#f59e0b`)
2. Sidebar: runs, noise sigma, methodology note
3. Main 3:1 column split:
   - Right (1): Strategy guide info panel (always visible)
   - Left (3): Run button → progress → results
4. [After run] Method cards × 3 (baseline / heuristic / RL Agent)
5. Relative comparison bars section
6. Charts row (avg emissions bar + per-run line chart)
7. System log

---

## 7. SVG Icon Patterns

Replace all emoji in the codebase with inline SVG. Size consistently at `20×20`.

**Pipeline icons (replaces 📡 🧮 🤖 ✅):**
```html
<!-- Grid / API -->
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>

<!-- Carbon Intelligence / compute -->
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>

<!-- RL Scheduling Engine -->
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>

<!-- Optimal Window / check -->
<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#22c55e" stroke-width="1.5" stroke-linecap="round"><polyline points="20 6 9 17 4 12"/></svg>
```

**Feature card icons** — use Lucide icon set stroke style, `stroke="#94a3b8"`, `stroke-width="1.5"`.  
Do not use fill icons or colorful icons for feature cards; they should be monochromatic.

---

## 8. Animation Rules

| Element | Allowed | Forbidden |
|---------|---------|-----------|
| Button hover | `transition: all 0.2s` color/shadow shift | Scale transform, bounce |
| Card hover | Border color fade, subtle `translateY(-2px)` | Scale, glow pulse |
| Nav brand dot | None (remove the pulsing `@keyframes pulse` animation) | Any continuous loop animation |
| Charts | Plotly's built-in render animation | Custom JS animation loops |
| Progress bar | Streamlit native | Fake indefinite spinners on finished results |
| Grid overlay | Static CSS pattern | Animated moving grid |
| Ticker / scrolling | Not used | Scrolling text marquee of any kind |

**`prefers-reduced-motion` compliance:**
```css
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        transition-duration: 0.01ms !important;
    }
}
```
Add this to every page's CSS block.

---

## 9. Plotly Chart Standards

| Property | Value |
|----------|-------|
| Background | `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"` |
| Font | `font=dict(family="IBM Plex Mono", size=10, color="#475569")` |
| Grid lines | `gridcolor="rgba(255,255,255,0.04)"` — y-axis only, x-axis off |
| Hover | `hovermode="x unified"` |
| Legend | `bgcolor="rgba(0,0,0,0)"`, IBM Plex Mono 10px `#475569` |
| Margins | `margin=dict(l=0, r=10, t=10, b=10)` |

**Chart type mapping:**
| Data | Chart type | Notes |
|------|-----------|-------|
| 24-hr carbon curve | Line + area fill | Actual=green solid, forecast=sky dashed, shade optimal window |
| Emissions comparison | Vertical bar | 2 bars: immediate=red, optimised=green |
| Strategy comparison | Vertical bar | 3 bars: baseline=red, heuristic=amber, RL=green |
| Per-run breakdown | Multi-line scatter | One line per strategy |
| Confidence decay | Not currently charted | If added: line with shaded uncertainty band |

---

## 10. Anti-Patterns (Do Not Use)

These patterns are present in the current codebase and should be removed or avoided in new work:

| Pattern | Location | Why it's wrong | Fix |
|---------|----------|---------------|-----|
| Scrolling ticker bar | `app.py` lines 335–354 | Fake "live" data, crypto aesthetic | Remove entirely; replace with static key-stats row |
| Pulsing `.brand-dot` animation | `app.py` nav + hero | Continuous animation with no information value | Remove `@keyframes pulse`; use a static dot |
| Emoji pipeline icons (`📡 🧮 🤖 ✅`) | `app.py` pipeline section | Inconsistent rendering, unprofessional register | Replace with inline SVG (see §7) |
| Emoji feature icons (`⚖️ 📉 🎲 🧠 🧪 📦`) | `app.py` feature cards | Same as above | Replace with Lucide SVG or remove decorative icons |
| `hero-h1` at 62px | `app.py` | Landing-page scale, not research tool | Reduce to 38–44px |
| `hero-eyebrow` pill with pulsing dot | `app.py` | Looks like a SaaS "Now available" badge | Replace with plain monospace citation line |
| Emoji in empty state placeholders | `overview.py`, `simulation.py` | Breaks icon consistency | Use a low-opacity SVG or text symbol |
| `text-shadow: 0 0 28px rgba(34,197,94,0.25)` on large numbers | `app.py` impact row | Neon glow effect — too hype-y | Remove; plain color is sufficient |
| `box-shadow: 0 0 12px rgba(34,197,94,0.7)` on brand dot | `app.py` | Excessive glow | Remove |

---

## 11. Streamlit-Specific Rules

```python
st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded",   # dashboard pages
    # initial_sidebar_state="collapsed", # landing page (app.py)
)
```

```css
/* Required on every page */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem !important; max-width: 100% !important; }

/* Landing page variant */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Plotly transparent backgrounds */
.js-plotly-plot .plotly .bg { fill: transparent !important; }

/* Progress bar (simulation page) */
div[data-testid="stProgress"] > div { background: rgba(34,197,94,0.12) !important; }
div[data-testid="stProgress"] > div > div { background: #22c55e !important; }
```

---

## 12. Pre-Delivery Checklist

Before committing any page change:

- [ ] No emoji used as icons anywhere in the page
- [ ] No scrolling ticker or continuous-loop "live" animation
- [ ] IBM Plex Mono used for all numeric values, labels, timestamps, badges
- [ ] IBM Plex Sans used for all prose/description text
- [ ] `prefers-reduced-motion` block included in CSS
- [ ] All Plotly charts use transparent backgrounds and IBM Plex Mono font
- [ ] Semantic color assignment is consistent (green=savings, red=baseline, amber=heuristic, sky=forecast)
- [ ] No `text-shadow` glow effects on metric numbers
- [ ] Page badge color matches §2.5 assignment
- [ ] KPI card variant matches page type (§5.2)
- [ ] System log uses `#040810` background and IBM Plex Mono 12px
