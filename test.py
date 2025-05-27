import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# Inverse transform the predictions
df = pd.read_csv("Company_stock_price.csv")
# Display the first few rows of the DataFrame   

df1 = df[:-5]
df2 = df[:-4]
df3 = df[:-3]
df4 = df[:-2]
df5 = df[:-1]

model = load_model(r"D:\Browns\CSE ANALYSER\best_lstm_model.h5")
print("Model loaded successfully.")

df1['Trade Date'] = pd.to_datetime(df1['Trade Date'])
df1 = df1.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df1['SMA_10'] = df1['Close (Rs.)'].rolling(10).mean()
df1['EMA_10'] = df1['Close (Rs.)'].ewm(span=10).mean()
df1['Momentum'] = df1['Close (Rs.)'] - df1['Close (Rs.)'].shift(10)
df1['Volatility'] = df1['Close (Rs.)'].rolling(10).std()
df1.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df1[feature_cols])
y_target = target_scaler.fit_transform(df1[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
future_predictions_scaled = []

for _ in range(5):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df1['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 6)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df11 = pd.concat([df1[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df11 = df11.sort_values('Trade Date').reset_index(drop=True)

###################################################################################################
#############################################
############################################################################################


df2['Trade Date'] = pd.to_datetime(df2['Trade Date'])
df2 = df2.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df2['SMA_10'] = df2['Close (Rs.)'].rolling(10).mean()
df2['EMA_10'] = df2['Close (Rs.)'].ewm(span=10).mean()
df2['Momentum'] = df2['Close (Rs.)'] - df2['Close (Rs.)'].shift(10)
df2['Volatility'] = df2['Close (Rs.)'].rolling(10).std()
df2.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df2[feature_cols])
y_target = target_scaler.fit_transform(df2[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
future_predictions_scaled = []

for _ in range(4):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df2['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 5)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df22 = pd.concat([df2[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df22 = df22.sort_values('Trade Date').reset_index(drop=True)

###################################################################################################
#############################################
############################################################################################


df3['Trade Date'] = pd.to_datetime(df3['Trade Date'])
df3 = df3.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df3['SMA_10'] = df3['Close (Rs.)'].rolling(10).mean()
df3['EMA_10'] = df3['Close (Rs.)'].ewm(span=10).mean()
df3['Momentum'] = df3['Close (Rs.)'] - df3['Close (Rs.)'].shift(10)
df3['Volatility'] = df3['Close (Rs.)'].rolling(10).std()
df3.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df3[feature_cols])
y_target = target_scaler.fit_transform(df3[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
future_predictions_scaled = []

for _ in range(3):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df3['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 4)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df33 = pd.concat([df3[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df33 = df33.sort_values('Trade Date').reset_index(drop=True)

###################################################################################################
#############################################
############################################################################################


df4['Trade Date'] = pd.to_datetime(df4['Trade Date'])
df4 = df4.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df4['SMA_10'] = df4['Close (Rs.)'].rolling(10).mean()
df4['EMA_10'] = df4['Close (Rs.)'].ewm(span=10).mean()
df4['Momentum'] = df4['Close (Rs.)'] - df4['Close (Rs.)'].shift(10)
df4['Volatility'] = df4['Close (Rs.)'].rolling(10).std()
df4.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df4[feature_cols])
y_target = target_scaler.fit_transform(df4[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
future_predictions_scaled = []

for _ in range(2):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df4['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 3)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df44 = pd.concat([df4[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df44 = df44.sort_values('Trade Date').reset_index(drop=True)

###################################################################################################
#############################################
############################################################################################


df5['Trade Date'] = pd.to_datetime(df5['Trade Date'])
df5 = df5.sort_values('Trade Date').reset_index(drop=True)

# 2. Create technical indicators
df5['SMA_10'] = df5['Close (Rs.)'].rolling(10).mean()
df5['EMA_10'] = df5['Close (Rs.)'].ewm(span=10).mean()
df5['Momentum'] = df5['Close (Rs.)'] - df5['Close (Rs.)'].shift(10)
df5['Volatility'] = df5['Close (Rs.)'].rolling(10).std()
df5.dropna(inplace=True)

# 3. Scale features
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df5[feature_cols])
y_target = target_scaler.fit_transform(df5[['Close (Rs.)']])

# 4. Prepare sequence
seq_length = 60
latest_sequence = X_features[-seq_length:]
latest_sequence = np.expand_dims(latest_sequence, axis=0)

# 5. Load model and predict
future_predictions_scaled = []

for _ in range(1):
    pred_scaled = model.predict(latest_sequence, verbose=0)[0]
    future_predictions_scaled.append(pred_scaled)

    # Dummy update of sequence (real implementation should use generated features)
    new_feature = latest_sequence[0, -1, :]
    latest_sequence = np.append(latest_sequence[:, 1:, :], [[new_feature]], axis=1)

# 6. Inverse transform
future_predictions = target_scaler.inverse_transform(np.array(future_predictions_scaled)).flatten()

# 7. Prepare forecast DataFrame
last_date = pd.to_datetime(df5['Trade Date'].iloc[-1])
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 2)]

future_df = pd.DataFrame({
    'Trade Date': future_dates,
    'Close (Rs.)': future_predictions
})

# 8. Concatenate with original
df55 = pd.concat([df5[['Trade Date', 'Close (Rs.)']], future_df], ignore_index=True)
df55 = df55.sort_values('Trade Date').reset_index(drop=True)

###################################################################################################
#############################################
############################################################################################

print("Forecasting completed for all datasets.")

plt.figure(figsize=(14, 7))
plt.plot(df['Trade Date'], df['Close (Rs.)'], label='Original', color='black', linewidth=2)
plt.plot(df11['Trade Date'], df11['Close (Rs.)'], label='Forecast (df11)', linestyle='--')
plt.plot(df22['Trade Date'], df22['Close (Rs.)'], label='Forecast (df22)', linestyle='--')
plt.plot(df33['Trade Date'], df33['Close (Rs.)'], label='Forecast (df33)', linestyle='--')
plt.plot(df44['Trade Date'], df44['Close (Rs.)'], label='Forecast (df44)', linestyle='--')
plt.plot(df55['Trade Date'], df55['Close (Rs.)'], label='Forecast (df55)', linestyle='--')
plt.xlabel('Trade Date')
plt.ylabel('Close (Rs.)')
plt.title('Stock Price Forecasts')
plt.legend()
plt.tight_layout()
plt.show()