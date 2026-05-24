import requests
import pandas as pd
from datetime import datetime


BASE_URL = "https://api.carbonintensity.org.uk"


class CarbonAPI:
    """
    Fetches UK National Grid Carbon Intensity data
    """

    def get_current_intensity(self):
        url = f"{BASE_URL}/intensity"
        response = requests.get(url)
        data = response.json()

        intensity = data["data"][0]["intensity"]["actual"]
        return intensity

    def get_forecast(self):
        url = f"{BASE_URL}/intensity/date"
        response = requests.get(url)
        data = response.json()

        forecast_data = []

        for item in data["data"]:
            forecast_data.append({
                "from": item["from"],
                "to": item["to"],
                "forecast": item["intensity"]["forecast"],
                "actual": item["intensity"]["actual"],
                "index": item["intensity"]["index"]
            })

        df = pd.DataFrame(forecast_data)
        df["from"] = pd.to_datetime(df["from"])
        df["to"] = pd.to_datetime(df["to"])

        return df

    def get_24h_forecast(self):
        df = self.get_forecast()

        now = pd.Timestamp.utcnow()
        next_24h = df[df["from"] <= now + pd.Timedelta(hours=24)]

        return next_24h.reset_index(drop=True)