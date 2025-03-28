import requests
import pandas as pd
import time
import json
from rpc_handler import ThingsBoardRPC  # Import the RPC handler

# Configuration
TB_CLOUD_URL = "https://thingsboard.cloud"
DEVICE_ACCESS_TOKEN = "4ji4dhuuufdcabhwl6ww"
TELEMETRY_INTERVAL = 10  # seconds

def send_telemetry(timestamp, temperature, humidity, co2, light, soil_moisture):
    telemetry = {
        "ts": int(pd.to_datetime(timestamp).timestamp() * 1000),
        "values": {
            "temperature": temperature,
            "humidity": humidity,
            "co2": co2,
            "light": light,
            "soil_moisture": soil_moisture
        }
    }
    
    url = f"{TB_CLOUD_URL}/api/v1/{DEVICE_ACCESS_TOKEN}/telemetry"
    response = requests.post(url, headers=headers, data=json.dumps(telemetry))
    
    if response.status_code == 200:
        print(f"Telemetry sent: {timestamp}")
    else:
        print(f"Telemetry failed ({response.status_code}): {response.text}")

if __name__ == "__main__":
    headers = {"Content-Type": "application/json"}
    
    # Initialize RPC handler
    rpc_service = ThingsBoardRPC(TB_CLOUD_URL, DEVICE_ACCESS_TOKEN)
    rpc_service.start()  # Runs in background thread
    
    # Main telemetry loop
    df = pd.read_csv("../data_generation/greenhouse_data.csv")
    for _, row in df.iterrows():
        send_telemetry(
            timestamp=row["timestamp"],
            temperature=row["temperature"],
            humidity=row["humidity"],
            co2=row["co2_ppm"],
            light=row["light_lux"],
            soil_moisture=row["soil_moisture"]
        )
        time.sleep(TELEMETRY_INTERVAL)
    
    rpc_service.stop()