import sys
import os
# Add the parent directory (Carbon-Aware-MLOps) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.simulator import MLTrainingSimulator

sim = MLTrainingSimulator()

runtime = sim.simulate_training(duration_minutes=1)
energy = sim.estimate_energy_kwh(runtime)

print(f"Runtime: {runtime:.2f} sec")
print(f"Energy: {energy:.4f} kWh")