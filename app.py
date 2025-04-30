import streamlit as st
import base64
import os

# Function to encode image to base64
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Function to set background image
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
    [data-testid="stTextInput"] > div > input {{
        background-color: rgba(255, 255, 255, 0.8);
        color: black;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the background image using a raw string for the file path
set_background(r"D:\Browns\CSE ANALYSER\backgrpund.jpg")  # Ensure the file exists at this path

# Streamlit app title
st.title("CSE ANALYSER")

# Input for company symbol
company_symbol = st.text_input("Enter Company Symbol:", "")

if company_symbol:
    st.write(f"You entered: {company_symbol}")
    # Placeholder for further processing
    st.write("Processing data for the company symbol...")

# Run ETL_PIPELINE.py with the company_symbol as a parameter
etl_command = f'python D:\Browns\CSE ANALYSER\ETL_PIPELINE.py "{company_symbol}"'
os.system(etl_command)