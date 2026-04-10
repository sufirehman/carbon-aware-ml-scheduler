import sys
import os
# Add the parent directory (Carbon-Aware-MLOps) to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.carbon_api import CarbonAPI
from core.scheduler import CarbonScheduler

api = CarbonAPI()
df = api.get_24h_forecast()

scheduler = CarbonScheduler(df)

best, worst, all_windows = scheduler.find_optimal_window(
    duration_minutes=120,
    urgency="medium"
)

print("Best Window:")
print(best)

print("\nWorst Window:")
print(worst)