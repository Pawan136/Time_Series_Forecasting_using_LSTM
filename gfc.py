import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
from datetime import date
import yfinance as yf

# Configuration
TICKER = "GOOGL"
START_DATE = "2000-01-01"
END_DATE = date.today().strftime("%Y-%m-%d")

WINDOW = 60
BATCH = 32
EPOCHS = 50
LR = 5e-4
PATIENCE = 6

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data loading
df = yf.download(TICKER, start=START_DATE, end=END_DATE, auto_adjust=True)
prices = df["Open"].values

# log returns
returns = np.log(prices[1:] / prices[:-1]).reshape(-1, 1)

# 60-20-20 split
n = len(returns)
train_end = int(n * 0.6)
val_end = int(n * 0.8)

train_r = returns[:train_end]
val_r = returns[train_end:val_end]
test_r = returns[val_end:]

# Scaling
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_r)
val_scaled = scaler.transform(val_r)
test_scaled = scaler.transform(test_r)

# Sequence builder
def make_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)


X_train, y_train = make_sequences(train_scaled, WINDOW)
X_val, y_val = make_sequences(val_scaled, WINDOW)
X_test, y_test = make_sequences(test_scaled, WINDOW)

X_train = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
y_train = torch.tensor(y_train, dtype=torch.float32, device=DEVICE)
X_val = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
y_val = torch.tensor(y_val, dtype=torch.float32, device=DEVICE)
X_test = torch.tensor(X_test, dtype=torch.float32, device=DEVICE)
y_test = torch.tensor(y_test, dtype=torch.float32, device=DEVICE)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH, shuffle=False)

# LSTM model
class ReturnLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1, 32, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


model = ReturnLSTM().to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Training with early stopping
best_val = float("inf")
patience = 0

for epoch in range(EPOCHS):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            val_loss += criterion(model(xb), yb).item()
    val_loss /= len(val_loader)

    if val_loss < best_val:
        best_val = val_loss
        patience = 0
        best_state = model.state_dict()
    else:
        patience += 1
        if patience >= PATIENCE:
            break

model.load_state_dict(best_state)

# Evaluation (CORRECT R²)
model.eval()
with torch.no_grad():
    preds_scaled = model(X_test).cpu().numpy()

preds = scaler.inverse_transform(preds_scaled)
true = test_r[WINDOW:]

rmse = np.sqrt(mean_squared_error(true, preds))
r2 = r2_score(true, preds)

print(f"RMSE (returns): {rmse:.6f}")
print(f"R2 score (returns): {r2:.4f}")   # should be ~0.01–0.05

# 7-day forecast (better than 30)
forecast_days = 7
last_window = test_scaled[-WINDOW:]
last_window = torch.tensor(last_window, dtype=torch.float32, device=DEVICE).unsqueeze(0)

future_returns = []

with torch.no_grad():
    for _ in range(forecast_days):
        next_ret = model(last_window).item()
        future_returns.append(next_ret)
        last_window = torch.cat(
            [last_window[:, 1:, :],
             torch.tensor([[[next_ret]]], device=DEVICE)],
            dim=1
        )

future_returns = scaler.inverse_transform(
    np.array(future_returns).reshape(-1, 1)
)

# Price reconstruction for plot
last_price = prices[-1]
future_prices = [last_price]

for r in future_returns:
    future_prices.append(future_prices[-1] * np.exp(r[0]))

future_prices = future_prices[1:]


# Plot
plt.figure(figsize=(14, 5))
plt.plot(df.index[-120:], df["Open"].iloc[-120:], label="historical")
future_dates = pd.date_range(df.index[-1], periods=forecast_days + 1)[1:]
plt.plot(future_dates, future_prices, label="7-day forecast", color="red")
plt.title("GOOGL 7-day price forecast (returns-based LSTM)")
plt.xlabel("date")
plt.ylabel("price")
plt.legend()
plt.grid(True)
plt.show()
