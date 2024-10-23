import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta

# Set the Streamlit page config
st.set_page_config(
    page_title="Prediction - Telecommunication",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.write("# Select Ticker to be predicted!")
companies = {
    "TLKM.JK": "PT Telkom Indonesia (Persero) Tbk",
    "ISAT.JK": "Indosat Tbk PT",
    "EXCL.JK": "XL Axiata Tbk PT"
}

# Initialize session state for last updated time and data storage if they don't exist
if 'last_update_time' not in st.session_state:
    st.session_state['last_update_time'] = datetime.now() - timedelta(minutes=30)  

if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}

# Function to fetch data from yfinance with caching mechanism
@st.cache_data(ttl=1800)
def fetch_data_yfinance(ticker_company, period_days, time_now):
    start_date = time_now - timedelta(days=period_days)
    data = yf.download(ticker_company, start=start_date, end=time_now, interval='1d')
    return data

# Function to check if the data is available for the current date
def is_data_available_for_today(data, time_now):
    if not data.empty and data.index[-1].date() == time_now.date():
        return True
    return False

# Prediction logic based on ticker
@st.cache_data(ttl=1800)
def predict_all(tickers, time_now):
    for ticker in tickers:
        st.write(f"## {companies[ticker]} Prediction Process")
        
        # Fetch data with caching
        data = fetch_data_yfinance(ticker, 30, time_now)
        
        # Store the fetched data in session_state
        st.session_state['cached_data'][ticker] = data

        # Check if data is available for today
        if is_data_available_for_today(data, time_now):
            st.dataframe(data)
            st.write(f"Data last updated at: {time_now.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            st.warning(f"Data for {ticker} on {time_now.strftime('%Y-%m-%d')} is not available yet.")

# Main app function
def main():
    # Get the current time
    current_time = datetime.now()
    
    # Check if 30 minutes have passed since the last update
    if (current_time - st.session_state['last_update_time']).total_seconds() >= 1800:
        st.write("Fetching new data for all companies...")
        st.session_state['last_update_time'] = current_time  # Update last update time
        
        # Run the prediction process for all companies
        predict_all(companies.keys(), current_time)
    else:
        time_left = 1800 - (current_time - st.session_state['last_update_time']).total_seconds()
        st.info(f"Next update in {int(time_left / 60)} minutes, last updated at {st.session_state['last_update_time']}")
        
        # Display cached data for the selected ticker
        ticker = st.selectbox('Which company do you want to predict?', list(companies.keys()))
        
        if ticker in st.session_state['cached_data']:
            data = st.session_state['cached_data'][ticker]
            if is_data_available_for_today(data, current_time):
                st.dataframe(data)
                st.write(f"Data last updated at: {st.session_state['last_update_time'].strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
