import base64
import shutil
import os

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    try:
        bin_str = get_base64_of_bin_file(jpg_file)
        print("✅ Background image loaded successfully.")
        # Note: No actual display in CLI. This is just for confirmation.
    except Exception as e:
        print(f"❌ Error loading background image: {e}")

# Set background (no actual effect in terminal, just for logging)
set_background(r"D:\Browns\CSE ANALYSER\backgrpund.jpg")

# Title
print("🏛️ CSE ANALYSER")

# Ask for input
company_symbol = input("Enter Company Symbol: ").strip().upper()

# Ask user to choose an action
print("\nChoose an action:")
print("1. 📈 Get Historical Data")
print("2. 📈 Forecast Stock Prices")
print("3. 🤓 Forecasting Comparison")
print("4. ⚠️ Risk Analysis (VaR)")

choice = input("Enter option number (1-4): ").strip()

if choice == "1":
    if company_symbol:
        local_folder = r"D:\Browns\CSE ANALYSER\Local csv of comp"
        target_file = r"D:\Browns\CSE ANALYSER\Company_stock_price.csv"
        source_file = os.path.join(local_folder, f"{company_symbol}.csv")

        if os.path.exists(source_file):
            shutil.copy(source_file, target_file)
            print(f"✅ Loaded local data for {company_symbol}")
            print("➡️ You can now run 'Get_Historical_Data.py'")
        else:
            print(f"❌ No local CSV found for {company_symbol}. Please check the symbol.")
    else:
        print("⚠️ Please enter a company symbol.")
elif choice == "2":
    print("➡️ Run 'Forecast.py' for Forecasting Stock Prices.")
elif choice == "3":
    print("➡️ Run 'Forcasing_comparisson.py' for Forecasting Comparison.")
elif choice == "4":
    print("➡️ Run 'var.py' for Risk Analysis (Value at Risk).")
else:
    print("❌ Invalid option selected.")
