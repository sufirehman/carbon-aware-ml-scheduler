import sys
import os
# Add the parent directory (Carbon-Aware-MLOps) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

st.set_page_config(
    page_title="Carbon-Aware ML Platform",
    layout="wide"
)

st.title("🌱 Carbon-Aware ML Training Platform")

st.markdown("""
Welcome 👋

This platform optimizes machine learning training schedules based on real-time carbon intensity.

### Navigate using the sidebar:
- 📊 Overview Dashboard
- 📈 Forecast Explorer
- ⚙️ Simulation Lab

---
Built for sustainable AI research & carbon-aware computing.
""")