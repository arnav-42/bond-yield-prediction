import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re

df = pd.read_csv('Data/yield-curve-rates-1990-2023.csv', parse_dates=['Date'])
df = df.sort_values('Date')

df = df.dropna(subset=['10 Yr'])
print(f"Dataset shape after dropping NaN values: {df.shape}")
y10 = df['10 Yr'].values

scaler = MinMaxScaler(feature_range=(0, 1))
y10_scaled = scaler.fit_transform(y10.reshape(-1, 1)).flatten()

#params
WINDOW_SIZE = 10
BATCH_SIZE = 32
N_AHEAD = 8

class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size):
        self.data = torch.FloatTensor(data)
        self.window_size = window_size
        
    def __len__(self):
        return len(self.data) - self.window_size
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]
        y = self.data[idx + self.window_size]
        return x, y

dataset = TimeSeriesDataset(y10_scaled, WINDOW_SIZE)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=WINDOW_SIZE, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = x.unsqueeze(-1)
        lstm_out, _ = self.lstm(x)
        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMModel().to(device)
loss_fn = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 100
losses = []

for epoch in range(epochs):
    epoch_loss = 0
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        y_pred = model(batch_x).squeeze()
        loss = loss_fn(y_pred, batch_y)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {avg_loss:.4f}')

plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()

model.eval()
new_data = torch.FloatTensor(y10_scaled[-WINDOW_SIZE:]).to(device)
preds_scaled = []

for _ in range(N_AHEAD):
    input_seq = new_data[-WINDOW_SIZE:].unsqueeze(0)  # [1, window_size]
    with torch.no_grad():
        pred = model(input_seq).item()
    preds_scaled.append(pred)
    new_data = torch.cat([new_data, torch.tensor([pred], device=device)])

# Inverse transform the predictions back to original scale
preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
print("Predictions for the next", N_AHEAD, "days:", preds)

# Plot the predictions
plt.figure(figsize=(12, 6))
# Use last 100 points of original data for visualization
plt.plot(df['Date'].values[-100:], y10[-100:], label='Historical Data')
last_date = df['Date'].iloc[-1]  # Get the last date in chronological order
pred_dates = pd.date_range(start=last_date, periods=N_AHEAD+1)[1:]  # Skip first date as it's the last historical date
plt.plot(pred_dates, preds, color='hotpink', label='Predictions')
plt.legend()
plt.title('10-Year Treasury Bond Yield Forecast')
plt.ylabel('Yield (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('yield_predictions.png')
plt.show()

# FINAL STATISTICS 
print("\n--- Final Model Statistics ---")
print("Final training loss (MAE):", losses[-1])

# Calculate performance on the most recent data
test_size = min(100, len(y10_scaled) - WINDOW_SIZE)
test_inputs = y10_scaled[-test_size-WINDOW_SIZE:-WINDOW_SIZE]
test_targets = y10_scaled[-test_size:]

with torch.no_grad():
    y_true = []
    y_pred = []
    
    for i in range(test_size):
        input_seq = torch.FloatTensor(test_inputs[i:i+WINDOW_SIZE]).unsqueeze(0).to(device)
        target = test_targets[i]
        
        prediction = model(input_seq).item()
        y_true.append(target)
        y_pred.append(prediction)
    
    # Convert to numpy arrays and reshape
    y_true = np.array(y_true).reshape(-1, 1)
    y_pred = np.array(y_pred).reshape(-1, 1)
    
    # Scale back to original range for interpretable metrics
    y_true_orig = scaler.inverse_transform(y_true).flatten()
    y_pred_orig = scaler.inverse_transform(y_pred).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(y_true_orig, y_pred_orig)
    mse = mean_squared_error(y_true_orig, y_pred_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_orig, y_pred_orig)
    
    print(f"Test MAE: {mae:.4f}")
    print(f"Test MSE: {mse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Show example predictions vs actual
    print("\nSample of last 5 predictions vs actual:")
    for i in range(-5, 0):
        print(f"Actual: {y_true_orig[i]:.4f}, Predicted: {y_pred_orig[i]:.4f}, Error: {abs(y_true_orig[i] - y_pred_orig[i]):.4f}")
        
# Plot actual vs predicted values for test data
plt.figure(figsize=(12, 6))
plt.plot(y_true_orig, label='Actual Values')
plt.plot(y_pred_orig, label='Predicted Values')
plt.title('Model Performance on Recent Data')
plt.xlabel('Time Step')
plt.ylabel('10-Year Treasury Yield (%)')
plt.legend()
plt.tight_layout()
plt.savefig('model_performance.png')
plt.show()
