import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def preprocess(data_path):
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df["temp_humidity_ratio"] = df["temperature"] / df["humidity"]
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    
    # Normalize
    scaler = MinMaxScaler()
    features = ["temperature", "humidity", "co2_ppm", "light_lux", "soil_moisture"]
    df[features] = scaler.fit_transform(df[features])
    
    # Split for ML
    X_train, X_test = train_test_split(df, test_size=0.2)
    return X_train, X_test, scaler