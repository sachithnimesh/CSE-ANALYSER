import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from matplotlib.animation import FuncAnimation

# 1. Load Dataset
df = pd.read_csv("Company_stock_price.csv")

# 🚨 2. Reorder Data: Oldest first (Ascending Trade Date)
df['Trade Date'] = pd.to_datetime(df['Trade Date'])  # ensure date type
df = df.sort_values('Trade Date').reset_index(drop=True)

# 3. Create Technical Indicators
df['SMA_10'] = df['Close (Rs.)'].rolling(10).mean()
df['EMA_10'] = df['Close (Rs.)'].ewm(span=10).mean()
df['Momentum'] = df['Close (Rs.)'] - df['Close (Rs.)'].shift(10)
df['Volatility'] = df['Close (Rs.)'].rolling(10).std()
df.dropna(inplace=True)

# 4. Scale Features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df[feature_cols])
y_target = target_scaler.fit_transform(df[['Close (Rs.)']])

# 5. Create Latest Sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]  # last 60 time steps
latest_sequence = np.expand_dims(latest_sequence, axis=0)  # reshape for LSTM input

# 6. Load Trained Model
model = load_model('best_lstm_model.h5')

# 7. Forecast 7 Future Days
future_predictions_scaled = []

for _ in range(7):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Update the sequence with the predicted value
    new_feature = latest_sequence[0, -1, :]  # repeat last feature (approximate)
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

future_predictions_scaled = np.array(future_predictions_scaled)

# 8. Inverse Scale to Actual Prices
future_predictions = target_scaler.inverse_transform(future_predictions_scaled).flatten()

# 9. Create Future Dates
last_date = pd.to_datetime(df['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

# 10. Create DataFrame for Forecasted Data
future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# Add empty columns for indicators (optional)
future_df['SMA_10'] = np.nan
future_df['EMA_10'] = np.nan
future_df['Momentum'] = np.nan
future_df['Volatility'] = np.nan

# 11. Append and Save
df_full = pd.concat([df, future_df], ignore_index=True)

# 🚨 IMPORTANT: Save final file again sorted by Trade Date ascending
df_full = df_full.sort_values('Trade Date').reset_index(drop=True)
df_full.to_csv("Company_stock_price_forecasted.csv", index=False)

print("✅ 7-Day Forecast Done and Saved to 'Company_stock_price_forecasted.csv'.")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('ggplot')  # safer alternative

fig, ax = plt.subplots(figsize=(10, 6))

def update(frame):
    ax.clear()
    ax.plot(df_full['Trade Date'][:frame], df_full['Close (Rs.)'][:frame], label='Close Price', color='blue')
    ax.set_title('Live Stock Price Forecast', fontsize=16)
    ax.set_xlabel('Trade Date', fontsize=12)
    ax.set_ylabel('Close Price (Rs.)', fontsize=12)
    ax.legend(loc='upper left')
    plt.xticks(rotation=45)

ani = FuncAnimation(fig, update, frames=len(df_full), interval=100, repeat=False)

plt.show()

# Plot Forecasted Dates
plt.figure(figsize=(12, 6))
plt.plot(df_full['Trade Date'], df_full['Close (Rs.)'], label='Historical Prices', color='blue')
plt.scatter(df_full['Trade Date'].iloc[-7:], df_full['Close (Rs.)'].iloc[-7:], label='Forecasted Prices', color='orange', marker='o')
plt.title('Stock Price Forecast', fontsize=16)
plt.xlabel('Trade Date', fontsize=12)
plt.ylabel('Close Price (Rs.)', fontsize=12)
plt.legend(loc='upper left')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()
