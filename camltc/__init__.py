# ── camltc/__init__.py ──
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np
import pandas as pd

from .carbon_api import CarbonAPI
from .scheduler import CarbonScheduler as UnderlingScheduler
from .rl_agent import RLScheduler as UnderlingRL

@dataclass
class ScheduleResult:
    best_window: str
    worst_window: str
    carbon_saving_pct: float
    carbon_saving_grams: float
    strategy: Literal["rl", "heuristic"]
    urgency: Literal["low", "medium", "high"]
    duration_minutes: int
    current_carbon_intensity: Optional[float] = None
    best_avg_carbon: Optional[float] = None
    worst_avg_carbon: Optional[float] = None


class CarbonScheduler:
    def __init__(self, duration_minutes=60, urgency="medium"):
        self.duration_minutes = duration_minutes
        self.urgency = urgency

    def recommend(self) -> ScheduleResult:
        """
        Main high-level orchestration method (Now clean and slim)
        """
        api = CarbonAPI()
        current = api.get_current_intensity()
        df = api.get_24h_forecast()

        # The Fix: Ensure 'carbon' exists globally across both pipelines
        df["carbon"] = df["actual"].fillna(df["forecast"])

        # 1. Run Baseline Heuristic
        heuristic = UnderlingScheduler(df)
        best, worst, _ = heuristic.find_optimal_window(
            duration_minutes=self.duration_minutes, urgency=self.urgency
        )

        # 2. Determine Strategy & Extract Best/Worst metrics
        strategy, best_start, best_end, best_avg = self._determine_strategy(df, best)
        worst_avg = float(worst["avg_carbon"])

        # 3. Compute Savings Outputs
        saving_pct, saving_grams = self._calculate_savings(best_avg, worst_avg)

        return ScheduleResult(
            best_window=f"{best_start} — {best_end}",
            worst_window=f"{worst['start']} — {worst['end']}",
            carbon_saving_pct=round(saving_pct, 1),
            carbon_saving_grams=round(saving_grams, 2),
            strategy=strategy,
            urgency=self.urgency,
            duration_minutes=self.duration_minutes,
            current_carbon_intensity=current,
            best_avg_carbon=round(best_avg, 1),
            worst_avg_carbon=round(worst_avg, 1),
        )

    def _determine_strategy(self, df: pd.DataFrame, best_heuristic: pd.Series):
        """Helper to evaluate data variance and fallback cleanly between RL and Heuristics"""
        carbon_values = df["carbon"].dropna().values
        volatility = float(np.std(carbon_values))

        if len(carbon_values) >= 4 and volatility > 30:
            try:
                rl = UnderlingRL(carbon_values)
                rl_index = rl.train()
                if rl_index is not None:
                    best_start = df["from"].iloc[rl_index]
                    best_end = df["to"].iloc[min(rl_index + 2, len(df)-1)]
                    best_avg = float(df["carbon"].iloc[rl_index:rl_index+2].mean())
                    return "rl", best_start, best_end, best_avg
            except:
                pass # Fallback cleanly to heuristic on any error

        return "heuristic", best_heuristic["start"], best_heuristic["end"], float(best_heuristic["avg_carbon"])

    def _calculate_savings(self, best_avg: float, worst_avg: float):
        """Helper to process mathematical formulas"""
        saving_pct = 0.0 if worst_avg == 0 else (worst_avg - best_avg) / worst_avg * 100
        duration_h = self.duration_minutes / 60
        saving_grams = (worst_avg - best_avg) * 0.3 * duration_h
        return saving_pct, saving_grams

__all__ = ["CarbonScheduler", "ScheduleResult"]
__version__ = "0.1.8"