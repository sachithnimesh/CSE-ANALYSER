import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 1. Load the Forecasted CSV
df = pd.read_csv("Company_stock_price_forecasted.csv")

# 2. Sort by Trade Date ascending (safety)
df['Trade Date'] = pd.to_datetime(df['Trade Date'])
df = df.sort_values('Trade Date').reset_index(drop=True)

# 3. Calculate Daily Returns
df['Returns'] = df['Close (Rs.)'].pct_change()
df.dropna(inplace=True)

# 4. Set Confidence Level
confidence_level = 0.95
alpha = 1 - confidence_level

#  5. VaR Methods

# (1) Historical Simulation VaR
hist_var = np.percentile(df['Returns'], 100 * alpha)

# (2) Parametric VaR (Assuming Normal Distribution)
mean_return = df['Returns'].mean()
std_return = df['Returns'].std()
parametric_var = norm.ppf(alpha, mean_return, std_return)

# (3) Monte Carlo Simulation VaR
# Simulate 10,000 random returns
np.random.seed(42)  # reproducibility
simulated_returns = np.random.normal(mean_return, std_return, 10000)
mc_var = np.percentile(simulated_returns, 100 * alpha)

# 6. Display Results
print("\n Value at Risk (1 Day, 95% Confidence):")
print(f"Historical Simulation VaR: {hist_var:.4f} ({hist_var*100:.2f}%)")
print(f"Parametric (Variance-Covariance) VaR: {parametric_var:.4f} ({parametric_var*100:.2f}%)")
print(f"Monte Carlo Simulation VaR: {mc_var:.4f} ({mc_var*100:.2f}%)")

# 7. Plot (Optional)
plt.figure(figsize=(10, 6))
plt.hist(df['Returns'], bins=50, color='skyblue', edgecolor='black', alpha=0.7, label='Historical Returns')
plt.axvline(hist_var, color='red', linestyle='--', label=f'Historical VaR ({hist_var:.2%})')
plt.axvline(parametric_var, color='green', linestyle='--', label=f'Parametric VaR ({parametric_var:.2%})')
plt.axvline(mc_var, color='purple', linestyle='--', label=f'Monte Carlo VaR ({mc_var:.2%})')
plt.title('Distribution of Returns and VaR')
plt.xlabel('Returns')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.show()

# 8. Function to interpret VaR results
def interpret_var(stock_price, historical_var, parametric_var, monte_carlo_var):
    """
    Interprets the Value at Risk (VaR) output in business terms.
    Args:
    - stock_price: The current stock price (used to calculate the potential loss in terms of value).
    - historical_var, parametric_var, monte_carlo_var: The VaR results in percentage.
    """
    print("\n📢 VaR Interpretation for 1-Day, 95% Confidence Level:")
    print(f"Current Stock Price: Rs. {stock_price}")
    
    def interpret_single_var(var_method, var_value):
        loss = stock_price * (var_value / 100)  # Calculate loss in Rs.
        print(f"\n{var_method} VaR Interpretation:")
        print(f"- Value at Risk (VaR): {var_value}%")
        print(f"- Potential loss: Rs. {loss:.2f}")
        print(f"- 95% confident that the stock will not lose more than Rs. {loss:.2f} in one day.")

    interpret_single_var("Historical Simulation", historical_var)
    interpret_single_var("Parametric (Variance-Covariance)", parametric_var)
    interpret_single_var("Monte Carlo Simulation", monte_carlo_var)

# Assuming you already have the forecasted data and actual VaR results:
stock_price = df['Close (Rs.)'].iloc[-1]  # Take the last forecasted stock price for interpretation

# Call the interpretation function
interpret_var(stock_price, hist_var, parametric_var, mc_var)
