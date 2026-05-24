"""
caml-tc — Carbon-Aware Machine Learning Training Controller
===========================================================
Schedule your ML training jobs to minimise carbon emissions
using real-time UK National Grid carbon intensity data.

Quickstart:
    from camltc import CarbonScheduler

    scheduler = CarbonScheduler(duration_minutes=90, urgency="low")
    result = scheduler.recommend()

    print(result.best_window)        # "03:00 — 04:30"
    print(result.carbon_saving_pct)  # 31.4
    print(result.strategy)           # "rl" or "heuristic"
"""

from __future__ import annotations

import sys
import os
from dataclasses import dataclass
from typing import Literal, Optional

# ── Make core/ importable whether installed as package or run from repo root ──
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler as _CoreScheduler
from core.rl_agent import RLScheduler as _CoreRL


# ── Public result dataclass ───────────────────────────────────────────────────

@dataclass
class ScheduleResult:
    """Returned by CarbonScheduler.recommend()"""

    best_window: str
    """Best execution window, e.g. '03:00 — 04:30 UTC'"""

    worst_window: str
    """Worst execution window — shows what you are avoiding"""

    carbon_saving_pct: float
    """Estimated % carbon saving vs worst (immediate) execution"""

    carbon_saving_grams: float
    """Estimated grams of CO₂ saved"""

    strategy: Literal["rl", "heuristic"]
    """Which scheduler produced this recommendation"""

    urgency: Literal["low", "medium", "high"]
    """Urgency level used"""

    duration_minutes: int
    """Workload duration used"""

    current_carbon_intensity: Optional[float] = None
    """Current grid carbon intensity in gCO₂/kWh (actual or forecast)"""

    best_avg_carbon: Optional[float] = None
    """Average carbon intensity in the best window (gCO₂/kWh)"""

    worst_avg_carbon: Optional[float] = None
    """Average carbon intensity in the worst window (gCO₂/kWh)"""

    def __str__(self) -> str:
        lines = [
            "─" * 48,
            "  CAML-TC — Carbon-Aware Scheduling Result",
            "─" * 48,
            f"  Best window      : {self.best_window}",
            f"  Worst window     : {self.worst_window}",
            f"  Carbon saving    : {self.carbon_saving_pct:.1f}%"
            f"  ({self.carbon_saving_grams:.2f} g CO₂)",
            f"  Strategy used    : {self.strategy}",
            f"  Urgency          : {self.urgency}",
            f"  Duration         : {self.duration_minutes} min",
        ]
        if self.current_carbon_intensity is not None:
            lines.append(
                f"  Grid now         : {self.current_carbon_intensity} gCO₂/kWh"
            )
        lines.append("─" * 48)
        return "\n".join(lines)


# ── Main public class ─────────────────────────────────────────────────────────

class CarbonScheduler:
    """
    Carbon-Aware ML Training Scheduler.

    Fetches real-time UK National Grid carbon intensity data and
    recommends the optimal execution window for your ML workload,
    using a heuristic scheduler or a trained RL agent depending
    on current grid volatility.

    Parameters
    ----------
    duration_minutes : int
        Expected training duration in minutes. Range: 30–240.
    urgency : str
        - ``"low"``    — aggressive carbon optimisation, longer delay allowed
        - ``"medium"`` — balanced carbon savings and delay (default)
        - ``"high"``   — run as soon as a clean-enough window appears

    Examples
    --------
    Basic recommendation::

        from camltc import CarbonScheduler

        scheduler = CarbonScheduler(duration_minutes=90, urgency="low")
        result = scheduler.recommend()
        print(result)

    Check just the best window::

        result = CarbonScheduler(duration_minutes=60).recommend()
        print(f"Run your job at: {result.best_window}")
        print(f"Carbon saving:   {result.carbon_saving_pct:.1f}%")

    High-urgency production job::

        result = CarbonScheduler(duration_minutes=30, urgency="high").recommend()
        print(result.best_window)
    """

    URGENCY_LEVELS = ("low", "medium", "high")
    MIN_DURATION = 30
    MAX_DURATION = 240
    # Grid volatility threshold above which RL outperforms heuristic
    _RL_VOLATILITY_THRESHOLD = 30.0  # gCO₂/kWh std dev

    def __init__(
        self,
        duration_minutes: int = 60,
        urgency: Literal["low", "medium", "high"] = "medium",
    ) -> None:
        if not self.MIN_DURATION <= duration_minutes <= self.MAX_DURATION:
            raise ValueError(
                f"duration_minutes must be {self.MIN_DURATION}–"
                f"{self.MAX_DURATION}, got {duration_minutes}"
            )
        if urgency not in self.URGENCY_LEVELS:
            raise ValueError(
                f"urgency must be one of {self.URGENCY_LEVELS}, got '{urgency}'"
            )
        self.duration_minutes = duration_minutes
        self.urgency = urgency

    def recommend(self) -> ScheduleResult:
        """
        Fetch live UK grid data and return a scheduling recommendation.

        Returns
        -------
        ScheduleResult
            Dataclass with best/worst windows, carbon savings estimate,
            and the strategy (rl or heuristic) that produced the result.

        Raises
        ------
        RuntimeError
            If the UK National Grid carbon intensity API is unreachable.
        ConnectionError
            If there is no internet connection.
        """
        import numpy as np

        # ── 1. Fetch live carbon data ─────────────────────────────────────
        try:
            api = CarbonAPI()
            current_intensity = api.get_current_intensity()
            df = api.get_24h_forecast()
        except Exception as exc:
            raise RuntimeError(
                "Could not reach the UK National Grid carbon intensity API. "
                "Check your internet connection. "
                f"Original error: {exc}"
            ) from exc

        if df.empty:
            raise RuntimeError(
                "UK National Grid API returned empty forecast data. "
                "Try again in a few minutes."
            )

        # ── 2. Run heuristic scheduler ────────────────────────────────────
        heuristic = _CoreScheduler(df)
        best_row, worst_row, _ = heuristic.find_optimal_window(
            duration_minutes=self.duration_minutes,
            urgency=self.urgency,
        )

        # ── 3. Decide whether to also run RL ─────────────────────────────
        carbon_values = df["carbon"].dropna().values
        volatility = float(np.std(carbon_values))
        use_rl = (
            len(carbon_values) >= 4
            and volatility > self._RL_VOLATILITY_THRESHOLD
        )

        strategy: Literal["rl", "heuristic"] = "heuristic"
        rl_exec_idx: Optional[int] = None

        if use_rl:
            try:
                rl_agent = _CoreRL(
                    carbon_values=carbon_values,
                    episodes=8000,
                    max_delay=20,
                )
                rl_exec_idx = rl_agent.train()
                strategy = "rl"
            except Exception:
                # RL failed — fall back to heuristic silently
                strategy = "heuristic"

        # ── 4. Determine best window from chosen strategy ─────────────────
        if strategy == "rl" and rl_exec_idx is not None:
            window_size = max(2, int(self.duration_minutes / 30))
            end_idx = min(rl_exec_idx + window_size, len(df) - 1)
            best_start = df["from"].iloc[rl_exec_idx]
            best_end = df["to"].iloc[end_idx]
            best_avg_carbon = float(
                df["carbon"].iloc[rl_exec_idx:end_idx + 1].mean()
            )
        else:
            best_start = best_row["start"]
            best_end = best_row["end"]
            best_avg_carbon = float(best_row["avg_carbon"])

        worst_start = worst_row["start"]
        worst_end = worst_row["end"]
        worst_avg_carbon = float(worst_row["avg_carbon"])

        # ── 5. Calculate carbon savings ───────────────────────────────────
        if worst_avg_carbon > 0:
            saving_pct = max(
                0.0,
                (worst_avg_carbon - best_avg_carbon) / worst_avg_carbon * 100,
            )
        else:
            saving_pct = 0.0

        # Estimate grams: E(kWh) ≈ P(kW) × T(h), assume ~0.3 kW avg GPU draw
        duration_hours = self.duration_minutes / 60
        estimated_kwh = 0.3 * duration_hours
        saving_grams = max(
            0.0,
            (worst_avg_carbon - best_avg_carbon) * estimated_kwh,
        )

        # ── 6. Format time strings ────────────────────────────────────────
        def _fmt(ts) -> str:
            try:
                return ts.strftime("%H:%M UTC")
            except Exception:
                return str(ts)

        return ScheduleResult(
            best_window=f"{_fmt(best_start)} — {_fmt(best_end)}",
            worst_window=f"{_fmt(worst_start)} — {_fmt(worst_end)}",
            carbon_saving_pct=round(saving_pct, 1),
            carbon_saving_grams=round(saving_grams, 2),
            strategy=strategy,
            urgency=self.urgency,
            duration_minutes=self.duration_minutes,
            current_carbon_intensity=current_intensity,
            best_avg_carbon=round(best_avg_carbon, 1),
            worst_avg_carbon=round(worst_avg_carbon, 1),
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def __repr__(self) -> str:
        return (
            f"CarbonScheduler("
            f"duration_minutes={self.duration_minutes}, "
            f"urgency='{self.urgency}')"
        )


# ── Package metadata ──────────────────────────────────────────────────────────
__version__ = "0.1.0"
__author__ = "Sufiyan Ul Rehman"
__email__ = "s.rehman@ulster.ac.uk"
__all__ = ["CarbonScheduler", "ScheduleResult"]
