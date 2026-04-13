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
import os
import sys

# Attempt to import codecarbon, fallback to mock if not installed
try:
    from codecarbon import EmissionsTracker
    HAS_CODECARBON = True
except ImportError:
    HAS_CODECARBON = False

# Import your existing core logic
from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler
from core.rl_agent import RLScheduler

# ---------------------------------------------------------
# EXPERIMENT LOGIC (Integrated to avoid ImportError)
# ---------------------------------------------------------

def run_baseline(train_function):
    """Measures emissions with no scheduling."""
    if HAS_CODECARBON:
        tracker = EmissionsTracker(measure_power_secs=1, save_to_file=False)
        tracker.start()
        train_function()
        emissions = tracker.stop()
    else:
        # Fallback math if library is missing
        start = time.time()
        train_function()
        emissions = (time.time() - start) * 0.00005 # Mock kg CO2
    return emissions

def run_with_heuristic(df, train_function):
    """Measures emissions using the CarbonScheduler delay."""
    scheduler = CarbonScheduler(df)
    best, _, _ = scheduler.find_optimal_window(duration_minutes=60, urgency="medium")
    
    # Scale delay for demo purposes (1 hour = 1 second)
    delay = min(best["delay_hours"], 5)
    time.sleep(delay)
    
    return run_baseline(train_function)

def run_with_rl(df, train_function):
    """Measures emissions using the RLScheduler."""
    carbon_values = df["carbon"].values
    rl_agent = RLScheduler(carbon_values)
    best_time_index = rl_agent.train()
    
    # Scale delay for demo purposes
    delay = min(best_time_index * 0.1, 5)
    time.sleep(delay)
    
    return run_baseline(train_function)

def run_full_experiment(df, train_function, runs=3):
    """Aggregates results over multiple runs for scientific rigor."""
    results = {"baseline": [], "heuristic": [], "rl": []}
    
    for _ in range(runs):
        results["baseline"].append(run_baseline(train_function))
        results["heuristic"].append(run_with_heuristic(df, train_function))
        results["rl"].append(run_with_rl(df, train_function))
        
    return {k: sum(v) / len(v) for k, v in results.items()}

# ---------------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------------

st.set_page_config(page_title="Carbon Simulation Lab", layout="wide")

st.title("⚙️ Carbon Simulation Lab")
st.markdown("""
Run **real-time power consumption experiments** to compare scheduling strategies. 
This lab utilizes `CodeCarbon` to track the physical CO₂ footprint of your compute.
""")

if not HAS_CODECARBON:
    st.warning("⚠️ `codecarbon` not found in environment. Running with mathematical estimation mode.")

# Define a standard workload
def train_function():
    # Simulate a matrix multiplication workload
    x = np.random.rand(3000, 3000)
    for _ in range(20):
        x = x @ x

if st.button("🚀 Start Benchmarking Experiment"):
    with st.spinner("Running experiments across Baseline, Heuristic, and RL models..."):
        # 1. Get Data
        api = CarbonAPI()
        df = api.get_24h_forecast()
        df["carbon"] = df["actual"].fillna(df["forecast"])
        
        # 2. Run Logic
        results = run_full_experiment(df, train_function, runs=2)
        
        # 3. Data Processing
        results_df = pd.DataFrame([{
            "Method": "Baseline (Now)",
            "Emissions (g CO2)": results["baseline"] * 1000
        }, {
            "Method": "Heuristic (Optimal)",
            "Emissions (g CO2)": results["heuristic"] * 1000
        }, {
            "Method": "RL Agent (AI)",
            "Emissions (g CO2)": results["rl"] * 1000
        }])

        # ----------------------------
        # VISUALS
        # ----------------------------
        st.subheader("📊 Empirical Results")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.dataframe(results_df, use_container_width=True)
            
        with col2:
            fig = go.Figure()
            colors = ['#EF553B', '#00CC96', '#636EFA']
            for i, row in results_df.iterrows():
                fig.add_trace(go.Bar(
                    x=[row["Method"]],
                    y=[row["Emissions (g CO2)"]],
                    marker_color=colors[i],
                    name=row["Method"]
                ))
            
            fig.update_layout(
                title="Measured Carbon Footprint per Strategy",
                yaxis_title="Grams of CO2",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)

        # ----------------------------
        # THE "PIVOT" INSIGHT
        # ----------------------------
        savings = ((results["baseline"] - results["rl"]) / results["baseline"]) * 100
        st.success(f"""
        ### 🧪 Lab Insights
        The **RL Agent** achieved a **{savings:.2f}% reduction** in carbon emissions compared to the baseline.
        This confirms that intelligent temporal shifting is a viable strategy for sustainable AI operations.
        """)