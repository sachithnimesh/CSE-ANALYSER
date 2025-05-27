import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

st.set_page_config(page_title="📉 VaR Risk Analysis", layout="wide")

st.title("⚠️ Value at Risk (VaR) - 1 Day Risk Estimation")
st.markdown("This page estimates how much your stock might lose **in a single day** with 95% confidence using different statistical methods.")

# 1. Load Forecasted CSV
df = pd.read_csv("Company_stock_price_forecasted.csv")

# 2. Prepare Data
df['Trade Date'] = pd.to_datetime(df['Trade Date'])
df = df.sort_values('Trade Date').reset_index(drop=True)

# 3. Calculate Daily Returns
df['Returns'] = df['Close (Rs.)'].pct_change()
df.dropna(inplace=True)

# 4. Confidence Level Setup
confidence_level = 0.95
alpha = 1 - confidence_level

# 5. VaR Calculations
# Historical Simulation
hist_var = np.percentile(df['Returns'], 100 * alpha)

# Parametric Method
mean_return = df['Returns'].mean()
std_return = df['Returns'].std()
parametric_var = norm.ppf(alpha, mean_return, std_return)

# Monte Carlo Simulation
np.random.seed(42)
simulated_returns = np.random.normal(mean_return, std_return, 10000)
mc_var = np.percentile(simulated_returns, 100 * alpha)

# 6. Display Raw Results
st.subheader("📊 Calculated VaR Results (95% Confidence, 1-Day Horizon)")
st.write(f"✅ Historical Simulation VaR: **{hist_var:.4f}** ({hist_var * 100:.2f}%)")
st.write(f"✅ Parametric (Variance-Covariance) VaR: **{parametric_var:.4f}** ({parametric_var * 100:.2f}%)")
st.write(f"✅ Monte Carlo Simulation VaR: **{mc_var:.4f}** ({mc_var * 100:.2f}%)")

# 7. Histogram Plot
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(df['Returns'], bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Historical Returns')
ax.axvline(hist_var, color='red', linestyle='--', label=f'Historical VaR ({hist_var:.2%})')
ax.axvline(parametric_var, color='green', linestyle='--', label=f'Parametric VaR ({parametric_var:.2%})')
ax.axvline(mc_var, color='purple', linestyle='--', label=f'Monte Carlo VaR ({mc_var:.2%})')
ax.set_title('Distribution of Returns and VaR', fontsize=14)
ax.set_xlabel('Returns')
ax.set_ylabel('Frequency')
ax.legend()
ax.grid(True)
st.pyplot(fig)

# 8. Business Interpretation
stock_price = df['Close (Rs.)'].iloc[-1]

st.subheader("📢 Interpretation for Business Decision Making")
st.markdown(f"""
**Current Stock Price:** Rs. `{stock_price:.2f}`

### 💡 What does this mean?

- 🕰 **1-Day Horizon**: We're estimating losses for the next trading day only.
- 🛡️ **95% Confidence Level**: We're 95% confident that losses won't exceed the VaR amount.

---

#### 📉 **Historical Simulation**
- VaR: `{hist_var:.2%}`
- Expected 1-day worst-case loss: **Rs. {stock_price * hist_var:.2f}**

#### 📊 **Parametric (Normal Distribution)**
- VaR: `{parametric_var:.2%}`
- Expected 1-day worst-case loss: **Rs. {stock_price * parametric_var:.2f}**

#### 🎲 **Monte Carlo Simulation**
- VaR: `{mc_var:.2%}`
- Expected 1-day worst-case loss: **Rs. {stock_price * mc_var:.2f}**

---

### 🧠 Final Takeaway:
> With 95% confidence, your stock is **not expected to lose more than Rs. {max(stock_price * hist_var, stock_price * parametric_var, stock_price * mc_var):.2f}** in one trading day under normal market conditions.

📝 But remember: Markets can be unpredictable. Always manage risk with stop-losses, diversification, and stress testing.
""")

# # Optional: Back button
# st.markdown("[🔙 Back to Home](../Home.py)")
