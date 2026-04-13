import pandas as pd
import numpy as np


class CarbonScheduler:
    def __init__(self, forecast_df):

        self.df = forecast_df.copy()

        # ✅ FIX: ensure datetime
        self.df["from"] = pd.to_datetime(self.df["from"])
        self.df["to"] = pd.to_datetime(self.df["to"])

        self.df["carbon"] = self.df["actual"].fillna(self.df["forecast"])

        self.df["hours_ahead"] = (
            self.df["from"] - pd.Timestamp.utcnow()
        ).dt.total_seconds() / 3600

        self.df["confidence"] = np.exp(-0.05 * self.df["hours_ahead"])

    def find_optimal_window(self, duration_minutes=60, urgency="medium"):

        # ✅ FIX: prevent zero window
        window_size = max(1, int(duration_minutes / 30))

        results = []

        for i in range(len(self.df) - window_size):
            window = self.df.iloc[i:i + window_size]

            weighted_carbon = np.average(
                window["carbon"],
                weights=window["confidence"]
            )

            delay_hours = window["hours_ahead"].iloc[0]
            penalty = self._urgency_penalty(delay_hours, urgency)

            score = weighted_carbon + penalty

            results.append({
                "start": window["from"].iloc[0],
                "end": window["to"].iloc[-1],
                "score": score,
                "avg_carbon": window["carbon"].mean(),
                "delay_hours": delay_hours
            })

        results_df = pd.DataFrame(results)

        # ✅ FIX: handle empty case
        if len(results_df) == 0:
            raise ValueError("No valid scheduling windows found.")

        best = results_df.loc[results_df["score"].idxmin()]
        worst = results_df.loc[results_df["score"].idxmax()]

        return best, worst, results_df

    def _urgency_penalty(self, delay_hours, urgency):

        if urgency == "low":
            factor = 0.1
        elif urgency == "medium":
            factor = 0.5
        else:
            factor = 1.5

        return factor * delay_hours