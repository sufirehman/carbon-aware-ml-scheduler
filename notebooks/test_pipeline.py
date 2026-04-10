import sys
import os
# Add the parent directory (Carbon-Aware-MLOps) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler
from core.simulator import MLTrainingSimulator

api = CarbonAPI()
df = api.get_24h_forecast()

scheduler = CarbonScheduler(df)

best, worst, _ = scheduler.find_optimal_window(
    duration_minutes=60,
    urgency="medium"
)

sim = MLTrainingSimulator()

runtime = sim.simulate_training(duration_minutes=1)
energy = sim.estimate_energy_kwh(runtime)

# Emissions
best_emissions = sim.calculate_emissions(energy, best["avg_carbon"])
worst_emissions = sim.calculate_emissions(energy, worst["avg_carbon"])

savings = ((worst_emissions - best_emissions) / worst_emissions) * 100

print(f"Best emissions: {best_emissions:.2f} gCO2")
print(f"Worst emissions: {worst_emissions:.2f} gCO2")
print(f"Carbon savings: {savings:.2f}%")