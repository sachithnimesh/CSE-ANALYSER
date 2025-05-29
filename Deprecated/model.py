import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt
import os


import pandas as pd
df = pd.read_csv('combined_final_data.csv')


# ====================== Step 1: Load & Clean Dataset ======================
print("Loading dataset...")
# df = pd.read_csv("combined_final_data.csv", dtype={"Symbol": "category"})
df['Trade Date'] = pd.to_datetime(df['Trade Date'])
df['Close (Rs.)'] = pd.to_numeric(df['Close (Rs.)'], errors='coerce')
df = df.dropna(subset=['Close (Rs.)'])
df = df.sort_values(by=['Symbol', 'Trade Date'])


# ====================== Step 2: Compute Indicators ======================
def compute_indicators(group):
    group['SMA_10'] = group['Close (Rs.)'].rolling(window=10).mean()
    group['EMA_10'] = group['Close (Rs.)'].ewm(span=10).mean()
    group['Momentum'] = group['Close (Rs.)'].diff(10)
    group['Volatility'] = group['Close (Rs.)'].rolling(window=10).std()
    return group

df = df.groupby('Symbol').apply(compute_indicators).reset_index(drop=True)
df = df.dropna()


# ====================== Step 3: Scale Features ======================
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']
target_col = 'Close (Rs.)'

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

df[feature_cols] = feature_scaler.fit_transform(df[feature_cols])
df[[target_col]] = target_scaler.fit_transform(df[[target_col]])

# ====================== Step 4: Create Sequences ======================
def create_sequences(data, feature_cols, target_col, seq_length=60):
    X, y = [], []
    for symbol, group in data.groupby('Symbol'):
        group = group.reset_index(drop=True)
        features = group[feature_cols].values
        target = group[target_col].values

        for i in range(len(group) - seq_length):
            X.append(features[i:i+seq_length])
            y.append(target[i+seq_length])
    return np.array(X), np.array(y)

print("Creating sequences...")
X, y = create_sequences(df, feature_cols, target_col, seq_length=60)

# ====================== Step 5: Train-Test Split ======================
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)

# ====================== Step 6: Build Model for Tuner ======================
def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(
        LSTM(units=hp.Int('units_layer1', min_value=32, max_value=256, step=32), return_sequences=True),
        input_shape=(X_train.shape[1], X_train.shape[2])
    ))
    model.add(Bidirectional(
        LSTM(units=hp.Int('units_layer2', min_value=32, max_value=256, step=32))
    ))
    model.add(Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
    model.add(Dense(1))
    model.compile(
        optimizer=Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
        loss=Huber(),
        metrics=['mae']
    )
    return model



# ====================== Step 7: Tune Hyperparameters ======================
print("Tuning hyperparameters...")
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=5,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='lstm_model'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
tuner.search(X_train, y_train, epochs=10, validation_data=(X_val, y_val), callbacks=[stop_early])



# ====================== Step 8: Evaluate Best Model ======================
best_model = tuner.get_best_models(num_models=1)[0]
y_pred_scaled = best_model.predict(X_val)
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled).flatten()
y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

print("\nModel Evaluation:")
print(f"MSE: {mean_squared_error(y_val_actual, y_pred_actual):.2f}")
print(f"MAE: {mean_absolute_error(y_val_actual, y_pred_actual):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_val_actual, y_pred_actual)):.2f}")
print(f"R2 Score: {r2_score(y_val_actual, y_pred_actual):.4f}")

# prompt: plot actual and validation data

import matplotlib.pyplot as plt

# Limit the plot to a reasonable number of data points to avoid memory issues or overcrowding
num_points_to_plot = 500

plt.figure(figsize=(15, 6))
plt.plot(y_val_actual[:num_points_to_plot], label='Actual Prices')
plt.plot(y_pred_actual[:num_points_to_plot], label='Predicted Prices')
plt.title('Actual vs. Predicted Stock Prices (Validation Data)')
plt.xlabel('Time Step (Scaled)')
plt.ylabel('Stock Price (Actual Scale)')
plt.legend()
plt.show()

# ====================== Step 9: Save the Model ======================
model_dir = 'saved_model'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
best_model.save(os.path.join(model_dir, 'best_lstm_model.h5'))
print(f"Model saved to {os.path.join(model_dir, 'best_lstm_model.h5')}")
