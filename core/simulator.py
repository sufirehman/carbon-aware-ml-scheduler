import time


class MLTrainingSimulator:
    def simulate_training(self, duration_minutes=10):
        """
        Simulates training by sleeping (lightweight + stable)
        """

        runtime_seconds = duration_minutes * 2  # fake compute scaling
        time.sleep(1)  # small delay to feel "real"

        return runtime_seconds

    def estimate_energy_kwh(self, runtime_seconds):
        """
        CPU-based estimation
        """

        power_kw = 0.065  # 65W CPU
        hours = runtime_seconds / 3600

        return power_kw * hours

    def calculate_emissions(self, energy_kwh, carbon_intensity):
        return energy_kwh * carbon_intensity