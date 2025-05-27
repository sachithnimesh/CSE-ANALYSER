import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

st.title("📈 7-Day Stock Price Forecast")

# File paths
csv_path = "D:/Browns/CSE ANALYSER/Company_stock_price.csv"
model_path = r"D:\Browns\CSE ANALYSER\best_lstm_model.h5"

# Check if files exist
if not os.path.exists(csv_path):
    st.error("❌ Stock price CSV file not found.")
    st.stop()
if not os.path.exists(model_path):
    st.error("❌ Trained LSTM model not found.")
    st.stop()

# 1. Load data
df = pd.read_csv(csv_path)
df['Trade Date'] = pd.to_datetime(df['Trade Date'])
df = df.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df['SMA_10'] = df['Close (Rs.)'].rolling(10).mean()
df['EMA_10'] = df['Close (Rs.)'].ewm(span=10).mean()
df['Momentum'] = df['Close (Rs.)'] - df['Close (Rs.)'].shift(10)
df['Volatility'] = df['Close (Rs.)'].rolling(10).std()
df.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df[feature_cols])
y_target = target_scaler.fit_transform(df[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
model = load_model(model_path)
future_predictions_scaled = []

for _ in range(7):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df_full = pd.concat([df[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df_full = df_full.sort_values('Trade Date').reset_index(drop=True)

# 9. Display Forecast
st.subheader("📅 Forecasted Prices")
st.dataframe(future_df)

# # 10. Plot
# fig, ax = plt.subplots(figsize=(12, 6))
# ax.plot(df['Trade Date'], df['Close (Rs.)'], label='Historical Prices', color='blue')
# ax.plot(future_df['Trade Date'], future_df['Close (Rs.)'], label='Forecasted Prices', color='orange', marker='o')
# ax.set_title('Stock Price Forecast (7 Days)')
# ax.set_xlabel('Trade Date')
# ax.set_ylabel('Close Price (Rs.)')
# ax.legend()
# plt.xticks(rotation=45)
# st.pyplot(fig)


# Assuming 'df' contains historical data and 'future_df' contains forecasted data
# Combine the datasets
df_combined = pd.concat([df, future_df])

# Create a new column to distinguish between historical and forecasted data
df_combined['Type'] = ['Historical'] * len(df) + ['Forecasted'] * len(future_df)

# Set 'Trade Date' as the index
df_combined.set_index('Trade Date', inplace=True)

# Pivot the data to have 'Historical' and 'Forecasted' as separate columns
df_pivot = df_combined.pivot(columns='Type', values='Close (Rs.)')

# Display the line chart
st.line_chart(df_pivot)


# 11. Optional: Save forecasted CSV
df_full.to_csv("D:/Browns/CSE ANALYSER/Company_stock_price_forecasted.csv", index=False)
st.success("✅ Forecast completed and saved!")

# st.markdown("[🔙 Back to Home](../Home.py)")
# st.markdown("[📊 Risk Analysis (VaR)](../var.py)")


# if st.button("🔙 Back to Home"):
#     st.session_state["page"] = "home"
#     st.experimental_rerun()