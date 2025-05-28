import streamlit as st
import base64
import streamlit.components.v1 as components
import subprocess

# --- Background Setup ---
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

# --- UI Layout ---
st.title("🏛️ CSE ANALYSER")

# Input and Button
company_symbol = st.text_input("Enter Company Symbol:", "")

if st.button("📈 Get Historical Data"):
    if company_symbol:
        st.session_state["symbol"] = company_symbol  # Store in session
        st.switch_page("pages/Get_Historical_Data.py")
    else:
        st.warning("Please enter a company symbol.")

if st.button("📈 Forecast Stock Prices"):
    st.switch_page("pages/Forecast.py")

if st.button("🤓 Forcsting Comparison"):
    st.switch_page("pages/Forcasing_comparisson.py")
if st.button("⚠️ Risk Analysis (VaR)"):
    st.switch_page("pages/var.py")