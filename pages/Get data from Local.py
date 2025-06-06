import streamlit as st
import os
import shutil

# # --- Background Setup ---
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_background(jpg_file):
#     try:
#         bin_str = get_base64_of_bin_file(jpg_file)
#         page_bg_img = f'''
#         <style>
#         .stApp {{
#             background-image: url("data:image/jpeg;base64,{bin_str}");
#             background-size: cover;
#             background-position: center;
#             background-repeat: no-repeat;
#             background-attachment: fixed;
#         }}
#         </style>
#         '''
#         st.markdown(page_bg_img, unsafe_allow_html=True)
#     except Exception as e:
#         st.warning(f"Error loading background: {e}")

# # Set background
# set_background(r"D:\Browns\CSE ANALYSER\backgrpund.jpg")

# Title
st.title("📁 Save Local CSV for Selected Company")

# --- Get company symbol from session state ---
symbol = st.session_state.get("symbol", "")

if not symbol:
    st.error("❌ No company symbol found in session. Please return to Home page and enter a symbol.")
    st.stop()

st.markdown(f"🔍 Looking for local CSV file for **{symbol}**...")

# Define paths
local_folder = r"D:\Browns\CSE ANALYSER\Local csv of comp"
target_file = r"D:\Browns\CSE ANALYSER\Company_stock_price.csv"
source_file = os.path.join(local_folder, f"{symbol}.csv")

# Attempt to copy file
if os.path.exists(source_file):
    shutil.copy(source_file, target_file)
    st.success(f"✅ Local data for **{symbol}** successfully copied to `Company_stock_price.csv`.")
    st.info("➡️ You can now proceed to 'Get_Historical_Data.py' or Forecast.")
else:
    st.error(f"❌ No local CSV found for **{symbol}**. Please check the file name in:\n`{local_folder}`")
