import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Detail",
    page_icon=":bulb:",
    layout="wide"
)

img_python_logo = Image.open('./pages/images/python-logo.png')
img_yahoo_finance = Image.open('./pages/images/yahoo-finance_BIG.png')

# Page title
st.title("Detail Of Implementation")
st.subheader("Prediction of Telecommunications Sector Stocks in Indonesia")

# Logo section with Python and Yahoo finance logos
col1, col2, col3 = st.columns([2, 0.6, 0.6])
with col1:
    st.write("### The time range of data used")
    st.write("August 2019 - August 2024")    
with col2:
    st.image(img_python_logo, width=150) 
with col3:
    st.image(img_yahoo_finance, width=150) 

# List of companies
st.write("## Indeks LQ45")
companies = {
    "TLKM": "PT Telkom Indonesia (Persero) Tbk",
    "ISAT": "Indosat Tbk PT",
    "EXCL": "XL Axiata Tbk PT"
}

for ticker, name in companies.items():
    st.write(f" **{ticker}** - {name}")

# Train/Test split
st.write("#### Train/Test Split")
col3, col4 = st.columns([1, 1])
with col3:
    st.metric("Train Data", "80%")
with col4:
    st.metric("Test Data", "20%")

# Display diagram or image
st.write("### Description of Long Short Term Memory Implementation")

# Load and display diagram image
img_lstm_cell = Image.open("./pages/images/lstm-cell.png")
st.image(img_lstm_cell, caption="LSTM Cell Architecture")

# Footer
st.write("© 2024 All Rights Reserved.")