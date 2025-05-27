import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="LSTM Forecast", layout="wide")

# Title
st.title("📈 LSTM Forecast for Stock Prices (Interactive)")
st.write("Forecast stock prices using an LSTM model with different trimming levels and visualize using Streamlit's line chart.")

# Load data and model path directly
uploaded_file = pd.read_csv("D:/Browns/CSE ANALYSER/Company_stock_price.csv")
model_path = r"D:\Browns\CSE ANALYSER\best_lstm_model.h5"

# Assign loaded file to df
df = uploaded_file

# Load model
try:
    model = load_model(model_path)
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# Forecasting function
def forecast_with_trim(df, trim, steps, model):
    df_trimmed = df[:-trim]
    df_trimmed['Trade Date'] = pd.to_datetime(df_trimmed['Trade Date'])
    df_trimmed = df_trimmed.sort_values('Trade Date').reset_index(drop=True)

    df_trimmed['SMA_10'] = df_trimmed['Close (Rs.)'].rolling(10).mean()
    df_trimmed['EMA_10'] = df_trimmed['Close (Rs.)'].ewm(span=10).mean()
    df_trimmed['Momentum'] = df_trimmed['Close (Rs.)'] - df_trimmed['Close (Rs.)'].shift(10)
    df_trimmed['Volatility'] = df_trimmed['Close (Rs.)'].rolling(10).std()
    df_trimmed.dropna(inplace=True)

    feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_features = feature_scaler.fit_transform(df_trimmed[feature_cols])
    y_target = target_scaler.fit_transform(df_trimmed[['Close (Rs.)']])

    seq_length = 60
    if len(X_features) < seq_length:
        st.warning(f"⚠ Not enough data after trimming {trim} to form sequence.")
        return None

    latest_sequence = X_features[-seq_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    future_predictions_scaled = []
    for _ in range(steps):
        pred_scaled = model.predict(latest_sequence, verbose=0)[0]
        future_predictions_scaled.append(pred_scaled)

        new_feature = latest_sequence[0, -1, :]  # keep last frame
        latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

    future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

    last_date = df_trimmed['Trade Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]

    future_df = pd.DataFrame({
        'Trade Date': future_dates,
        'Close (Rs.)': future_predictions
    })

    result_df = pd.concat([df_trimmed[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
    result_df = result_df.sort_values('Trade Date').reset_index(drop=True)

    return result_df

# Sidebar
st.sidebar.header("Forecast Settings")
trim_steps = st.sidebar.slider("Select Max Trim Level", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast Days", 1, 10, 5)

# Run forecasts for different trims
results = []
for i in range(trim_steps, 0, -1):
    result = forecast_with_trim(df.copy(), i, forecast_days, model)
    if result is not None:
        result['Trim'] = f'Trim_{i}'
        results.append(result)

# Combine results for chart
if results:
    combined_df = pd.DataFrame()
    for result in results:
        label = result['Trim'].iloc[0]
        temp_df = result[['Trade Date', 'Close (Rs.)']].copy()
        temp_df = temp_df.rename(columns={'Close (Rs.)': label})
        if combined_df.empty:
            combined_df = temp_df
        else:
            combined_df = pd.merge(combined_df, temp_df, on='Trade Date', how='outer')

    combined_df = combined_df.sort_values('Trade Date').set_index('Trade Date')
    st.subheader("📈 Forecast Chart (using st.line_chart)")
    st.line_chart(combined_df)
else:
    st.warning("⚠ No valid forecasts generated.")
