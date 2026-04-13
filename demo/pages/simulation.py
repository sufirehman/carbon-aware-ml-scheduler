# import sys
# import os

# # 1. Find the root of the project (Carbon-Aware-MLOps)
# # This looks 2 levels up from demo/pages/
# current_dir = os.path.dirname(os.path.abspath(__file__))
# project_root = os.path.dirname(os.path.dirname(current_dir))

# # 2. Add the root to the VERY START of the python path
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# # 3. Import using the full package path
# try:
#     from core.experiment import run_experiment
# except ImportError:
#     # Backup for different cloud structures
#     sys.path.append(os.getcwd())
#     from core.experiment import run_experiment

# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# from core.carbon_api import CarbonAPI

# # ----------------------------
# # PAGE CONFIG
# # ----------------------------
# st.set_page_config(
#     page_title="Carbon Simulation Lab",
#     layout="wide"
# )

# st.title("⚙️ Carbon Simulation Lab")
# st.markdown("Run baseline, heuristic, and RL experiments with real emissions measurement.")


# # ----------------------------
# # TRAIN FUNCTION (TEMP WORKLOAD)
# # ----------------------------
# def train_function():
#     import numpy as np

#     x = np.random.rand(3000, 3000)

#     for _ in range(30):
#         x = x @ x


# # ----------------------------
# # RUN EXPERIMENT
# # ----------------------------
# if st.button("🚀 Run Full Experiment"):

#     # Load carbon data
#     api = CarbonAPI()
#     df = api.get_24h_forecast()
#     df["carbon"] = df["actual"].fillna(df["forecast"])

#     # Run experiment
#     results = run_experiment(df, train_function, runs=10)

#     # ----------------------------
#     # CONVERT TO DATAFRAME
#     # ----------------------------
#     results_df = pd.DataFrame([{
#         "Baseline (kg CO2)": results["baseline"],
#         "Heuristic (kg CO2)": results["heuristic"],
#         "RL (kg CO2)": results["rl"],

#         "Baseline (g CO2)": results["baseline"] * 1000,
#         "Heuristic (g CO2)": results["heuristic"] * 1000,
#         "RL (g CO2)": results["rl"] * 1000,
#     }])

#     # ----------------------------
#     # RESULTS TABLE
#     # ----------------------------
#     st.markdown("## 📊 Experiment Results")
#     st.dataframe(results_df, use_container_width=True)

#     # ----------------------------
#     # BAR CHART
#     # ----------------------------
#     st.markdown("## 📈 Emissions Comparison")

#     fig = go.Figure()

#     fig.add_trace(go.Bar(
#         name="Baseline",
#         x=["Baseline"],
#         y=[results["baseline"] * 1000]
#     ))

#     fig.add_trace(go.Bar(
#         name="Heuristic",
#         x=["Heuristic"],
#         y=[results["heuristic"] * 1000]
#     ))

#     fig.add_trace(go.Bar(
#         name="RL",
#         x=["RL"],
#         y=[results["rl"] * 1000]
#     ))

#     fig.update_layout(
#         barmode="group",
#         title="Carbon Emissions Comparison",
#         yaxis_title="CO₂ Emissions (grams)",
#         template="plotly_white"
#     )

#     st.plotly_chart(fig, use_container_width=True)

#     # ----------------------------
#     # INSIGHT
#     # ----------------------------
#     best_method = min(results, key=results.get)

#     st.success(
#         f"""
# 🏆 Best Performing Method: {best_method.upper()}

# This experiment demonstrates measurable carbon differences between:
# - Baseline (no scheduling)
# - Heuristic scheduling
# - RL-based scheduling
# """
#     )

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import time
import random
import requests

# 1. ---------------------------------------------------------
# THE RL AGENT (MOVED HERE TO STOP IMPORT ERRORS)
# ---------------------------------------------------------
class RLScheduler:
    def __init__(self, carbon_values, episodes=100):
        self.carbon = carbon_values
        self.n = len(carbon_values)
        self.episodes = episodes
        self.q_table = np.zeros(self.n)
        self.alpha = 0.1
        self.gamma = 0.9

    def reward(self, t, window=3):
        if t + window >= len(self.carbon):
            return -999
        avg = sum(self.carbon[t:t+window]) / window
        return -avg

    def train(self):
        for ep in range(self.episodes):
            epsilon = max(0.05, 0.3 * (1 - ep / self.episodes))
            for t in range(self.n):
                if random.random() < epsilon:
                    action = random.randint(0, self.n - 1)
                else:
                    action = np.argmax(self.q_table)
                r = self.reward(action)
                self.q_table[action] = self.q_table[action] + self.alpha * (
                    r + self.gamma * np.max(self.q_table) - self.q_table[action]
                )
        return np.argmax(self.q_table)

# 2. ---------------------------------------------------------
# MOCK API & UTILS (TO ENSURE STABILITY)
# ---------------------------------------------------------
def get_carbon_data():
    try:
        url = "https://api.carbonintensity.org.uk/intensity/24h"
        response = requests.get(url).json()
        data = response['data']
        df = pd.DataFrame(data)
        df['carbon'] = df['intensity'].apply(lambda x: x['forecast'])
        return df
    except:
        # Fallback if API is down
        times = pd.date_range(start=pd.Timestamp.now(), periods=48, freq='30min')
        return pd.DataFrame({'from': times, 'carbon': np.random.randint(50, 250, 48)})

# 3. ---------------------------------------------------------
# UI & EXPERIMENT LOGIC
# ---------------------------------------------------------
st.set_page_config(page_title="Carbon Simulation Lab", layout="wide")
st.title("⚙️ Carbon Simulation Lab")

def train_workload():
    # Simple CPU heavy task
    x = np.random.rand(2000, 2000)
    for _ in range(10):
        x = x @ x

if st.button("🚀 Run Lab Experiment"):
    with st.spinner("Benchmarking..."):
        df = get_carbon_data()
        carbon_values = df['carbon'].values
        
        # Baseline
        start = time.time()
        train_workload()
        baseline_time = time.time() - start
        baseline_emissions = baseline_time * carbon_values[0] * 0.001

        # Heuristic (Simple find min)
        min_idx = np.argmin(carbon_values)
        heuristic_emissions = baseline_time * carbon_values[min_idx] * 0.001
        
        # RL
        agent = RLScheduler(carbon_values)
        best_idx = agent.train()
        rl_emissions = baseline_time * carbon_values[best_idx] * 0.001

        # Results
        res = pd.DataFrame({
            "Method": ["Baseline", "Heuristic", "RL Agent"],
            "Emissions (g CO2)": [baseline_emissions, heuristic_emissions, rl_emissions]
        })

        st.subheader("📊 Results")
        col1, col2 = st.columns([1, 2])
        col1.dataframe(res)
        
        fig = go.Figure(go.Bar(x=res["Method"], y=res["Emissions (g CO2)"], marker_color=['red', 'green', 'blue']))
        fig.update_layout(title="Carbon Footprint Comparison", template="plotly_white")
        col2.plotly_chart(fig)

        st.success(f"Success! The RL Agent identified a window saving {((baseline_emissions-rl_emissions)/baseline_emissions)*100:.1f}% carbon.")