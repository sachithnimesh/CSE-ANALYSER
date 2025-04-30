import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
import keras_tuner as kt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# 1. Load Dataset
df = pd.read_csv("Company_stock_price.csv")

# 2. Create Technical Indicators
df['SMA_10'] = df['Close (Rs.)'].rolling(10).mean()
df['EMA_10'] = df['Close (Rs.)'].ewm(span=10).mean()
df['Momentum'] = df['Close (Rs.)'] - df['Close (Rs.)'].shift(10)
df['Volatility'] = df['Close (Rs.)'].rolling(10).std()
df.dropna(inplace=True)

# 3. Scale Features and Target Separately
feature_cols = ['SMA_10', 'EMA_10', 'Momentum', 'Volatility']  # 🚨 Removed Close price
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_features = feature_scaler.fit_transform(df[feature_cols])
y_target = target_scaler.fit_transform(df[['Close (Rs.)']])

# 4. Create Sequences
def create_sequences(features, target, seq_length):
    X, y = [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length])
        y.append(target[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60  # Longer memory
X, y = create_sequences(X_features, y_target, seq_length)

# 5. Train-Test Split (Chronologically, No Shuffle)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# 6. Build Model for Hyperparameter Tuning
def build_model(hp):
    model = Sequential()
    model.add(Bidirectional(LSTM(
        units=hp.Int('units1', 64, 256, step=32),
        return_sequences=True,
        activation='tanh'
    ), input_shape=(X_train.shape[1], X_train.shape[2])))
    
    model.add(Bidirectional(LSTM(
        units=hp.Int('units2', 32, 128, step=32),
        return_sequences=False,
        activation='tanh'
    )))
    
    model.add(Dropout(hp.Float('dropout', 0.2, 0.5, step=0.1)))
    model.add(Dense(1))

    model.compile(
        optimizer=Adam(learning_rate=hp.Choice('lr', [1e-3, 5e-4, 1e-4])),
        loss=Huber(),   # 🚨 Changed from MSE to Huber loss
        metrics=['mae']
    )
    return model

# 7. Hyperparameter Tuning
tuner = kt.RandomSearch(
    build_model,
    objective='val_loss',
    max_trials=30,   # Reduced to save time
    executions_per_trial=1,
    directory='tuned_model',
    project_name='bi_lstm_improved'
)

tuner.search(X_train, y_train, epochs=30, validation_data=(X_val, y_val), batch_size=32, verbose=1)

# 8. Train Best Model
best_hp = tuner.get_best_hyperparameters(1)[0]
model = tuner.hypermodel.build(best_hp)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
]

history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val),
                    batch_size=32, callbacks=callbacks, verbose=1)

# 9. Predict & Evaluate
y_pred_scaled = model.predict(X_val)
y_val_actual = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
y_pred_actual = target_scaler.inverse_transform(y_pred_scaled).flatten()

# 10. Metrics
mse = mean_squared_error(y_val_actual, y_pred_actual)
mae = mean_absolute_error(y_val_actual, y_pred_actual)
rmse = np.sqrt(mse)
r2 = r2_score(y_val_actual, y_pred_actual)

print(f"\n🔍 Evaluation on Validation Set:")
print(f"MSE: {mse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 Score: {r2:.4f}")

# 11. Plot
plt.figure(figsize=(12, 6))
plt.plot(y_val_actual, label='Actual Price', color='blue')
plt.plot(y_pred_actual, label='Predicted Price', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Price')
plt.xlabel('Time Step')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

# 12. Save Predictions
predictions_df = pd.DataFrame({
    'Actual': y_val_actual,
    'Predicted': y_pred_actual
})
predictions_df.to_csv('predictions.csv', index=False)
print("Predictions saved to 'predictions.csv'.")

# 13. Save the Trained Model
model.save('final_trained_model.h5')
print("Trained model saved as 'final_trained_model.h5'.")

