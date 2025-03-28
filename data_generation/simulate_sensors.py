import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_greenhouse_data(num_days=30):
    timestamps = [datetime.now() - timedelta(days=i) for i in range(num_days)]
    
    data = pd.DataFrame({
        "timestamp": timestamps * 24,  # Hourly data
        "temperature": np.concatenate([
            np.linspace(18, 28, 12),  # Daytime rise
            np.linspace(28, 16, 12)   # Nighttime fall
        ] * num_days) + np.random.normal(0, 1, num_days*24),
        "humidity": np.random.uniform(40, 80, num_days*24),
        "co2_ppm": np.random.normal(800, 150, num_days*24).clip(400, 2000),
        "light_lux": np.concatenate([
            np.zeros(6),              # Night
            np.linspace(0, 50_000, 6),  # Sunrise
            np.linspace(50_000, 0, 6),  # Sunset
            np.zeros(6)
        ] * num_days),
        "soil_moisture": np.random.uniform(30, 70, num_days*24)
    })
    return data

data = generate_greenhouse_data()
data.to_csv("data_generation/greenhouse_data.csv", index=False)