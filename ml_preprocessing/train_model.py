import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import numpy as np

# 1. Data Preprocessing
def preprocess(data_path, window_size=24):
    df = pd.read_csv(data_path)
    
    # Feature engineering
    df["temp_humidity_ratio"] = df["temperature"] / df["humidity"]
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    
    # Normalize
    scaler = MinMaxScaler()
    features = ["temperature", "humidity", "co2_ppm", "light_lux", "soil_moisture"]
    df[features] = scaler.fit_transform(df[features])
    
    # Convert to sequences (24-hour windows)
    def create_sequences(data, window_size):
        sequences = []
        for i in range(len(data) - window_size):
            seq = data[i:i+window_size]
            sequences.append(seq)
        return np.array(sequences)
    
    X = create_sequences(df[features].values, window_size)
    y = df[features].values[window_size:]  # Next step after each window
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, y_train, scaler

# 2. Define LSTM Model
class GreenhouseLSTM(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, output_size=5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)  # out shape: (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])  # Last timestep only
        return out

# 3. Training Function
def train_lstm(X_train, y_train, epochs=50):
    model = GreenhouseLSTM()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Convert to PyTorch tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), "ml_models/greenhouse_predictor.pth")
    return model

# 4. Execute Pipeline
if __name__ == "__main__":
    X_train, y_train, scaler = preprocess(
        "../data_generation/greenhouse_data.csv", 
        window_size=24  # Explicitly define window size
    )
    train_lstm(X_train, y_train)
    joblib.dump(scaler, "ml_models/scaler.pkl")