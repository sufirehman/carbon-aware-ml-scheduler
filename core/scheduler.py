import pandas as pd
import numpy as np


class CarbonScheduler:
    def __init__(self, forecast_df):
        """
        forecast_df: DataFrame with columns:
        ['from', 'to', 'forecast', 'actual']
        """
        self.df = forecast_df.copy()

        # Use forecast if actual is missing
        self.df["carbon"] = self.df["actual"].fillna(self.df["forecast"])

        # Confidence weighting (future = less reliable)
        self.df["hours_ahead"] = (
            self.df["from"] - pd.Timestamp.utcnow()
        ).dt.total_seconds() / 3600

        # Confidence decay (exponential)
        self.df["confidence"] = np.exp(-0.05 * self.df["hours_ahead"])

    def find_optimal_window(self, duration_minutes=60, urgency="medium"):
        """
        duration_minutes: training job duration
        urgency: low / medium / high
        """

        window_size = int(duration_minutes / 30)  # API is 30-min intervals

        results = []

        for i in range(len(self.df) - window_size):
            window = self.df.iloc[i:i + window_size]

            # Weighted carbon score
            weighted_carbon = np.average(
                window["carbon"],
                weights=window["confidence"]
            )

            # Delay penalty
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

        best = results_df.loc[results_df["score"].idxmin()]
        worst = results_df.loc[results_df["score"].idxmax()]

        return best, worst, results_df

    def _urgency_penalty(self, delay_hours, urgency):
        """
        Penalizes waiting too long
        """

        if urgency == "low":
            factor = 0.1
        elif urgency == "medium":
            factor = 0.5
        else:  # high
            factor = 1.5

        return factor * delay_hours