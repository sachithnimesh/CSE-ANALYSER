import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Load data and model
df = pd.read_csv("Company_stock_price.csv")
model = load_model(r"D:\Browns\CSE ANALYSER\best_lstm_model.h5")
print("Model loaded successfully.")

def forecast_with_trim(df, trim, steps, model):
    # 1. Trim the DataFrame
    df_trimmed = df[:-trim]
    
    # 2. Convert date and sort
    df_trimmed['Trade Date'] = pd.to_datetime(df_trimmed['Trade Date'])
    df_trimmed = df_trimmed.sort_values('Trade Date').reset_index(drop=True)

    # 3. Add technical indicators
    df_trimmed['SMA_10'] = df_trimmed['Close (Rs.)'].rolling(10).mean()
    df_trimmed['EMA_10'] = df_trimmed['Close (Rs.)'].ewm(span=10).mean()
    df_trimmed['Momentum'] = df_trimmed['Close (Rs.)'] - df_trimmed['Close (Rs.)'].shift(10)
    df_trimmed['Volatility'] = df_trimmed['Close (Rs.)'].rolling(10).std()
    df_trimmed.dropna(inplace=True)

    # 4. Scale features
    feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()

    X_features = feature_scaler.fit_transform(df_trimmed[feature_cols])
    y_target = target_scaler.fit_transform(df_trimmed[['Close (Rs.)']])

    # 5. Prepare sequence
    seq_length = 60
    if len(X_features) < seq_length:
        print(f"Not enough data for sequence length {seq_length} after trimming {trim}. Skipping.")
        return None

    latest_sequence = X_features[-seq_length:]
    latest_sequence = np.expand_dims(latest_sequence, axis=0)

    # 6. Forecast
    future_predictions_scaled = []
    for _ in range(steps):
        pred_scaled = model.predict(latest_sequence, verbose=0)[0]
        future_predictions_scaled.append(pred_scaled)

        # For demo: use last feature again (real case should generate new indicators)
        new_feature = latest_sequence[0, -1, :]
        latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

    # 7. Inverse transform
    future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

    # 8. Prepare forecast DataFrame
    last_date = df_trimmed['Trade Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, steps + 1)]

    future_df = pd.DataFrame({
        'Trade Date': future_dates,
        'Close (Rs.)': future_predictions
    })

    # 9. Concatenate
    result_df = pd.concat([df_trimmed[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
    result_df = result_df.sort_values('Trade Date').reset_index(drop=True)

    return result_df

# Run forecasts with different trimming levels and prediction steps
trim_steps = [(5, 5), (4, 4), (3, 3), (2, 2), (1, 1)]
results = []

for trim, steps in trim_steps:
    forecast_df = forecast_with_trim(df, trim, steps, model)
    if forecast_df is not None:
        results.append(forecast_df)

# Optional: plot or save results
for i, result_df in enumerate(results):
    plt.plot(result_df['Trade Date'], result_df['Close (Rs.)'], label=f"Trim {trim_steps[i][0]}")
plt.legend()
plt.title("LSTM Forecasts at Different Trimming Levels")
plt.xlabel("Date")
plt.ylabel("Close (Rs.)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
