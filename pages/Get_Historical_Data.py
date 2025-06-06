import streamlit as st
import pandas as pd
import subprocess
import os
import base64

# Background setup
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(jpg_file):
    bin_str = get_base64_of_bin_file(jpg_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/jpeg;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background(r"D:\Browns\CSE ANALYSER\backgrpund.jpg")

st.title("📊 Historical Stock Data Result")

# Get symbol from session
company_symbol = st.session_state.get("symbol", "")

if not company_symbol:
    st.warning("No company symbol found. Please go back to homepage.")
    st.stop()

st.write(f"Processing data for: **{company_symbol}**...")

etl_script_path = r"D:\Browns\CSE ANALYSER\ETL.py"

try:
    result = subprocess.run(["python", etl_script_path, company_symbol],
                            capture_output=True, text=True, check=True)
    st.success("✅ ETL process completed.")
    st.code(result.stdout)
except subprocess.CalledProcessError as e:
    st.error("❌ ETL process failed.")
    st.code(e.stderr)
    st.stop()

file_path = r"D:\Browns\CSE ANALYSER\Company_stock_price.csv"
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    st.subheader("📈 Company Stock Price Data")
    st.dataframe(df)
else:
    st.warning("⚠️ CSV file not found. Check ETL script output.")