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


#Logic function for prediction step
#fetch and store to cache data
@st.cache_data(ttl=1800)
def fetch_data_yfinance(ticker_company, period_days, time_now):
    try:
        ticker = yf.Ticker(ticker_company)
        start_date = time_now - timedelta(days=period_days)
        yesterday = time_now - timedelta(days=1)
        data = ticker.history(start=start_date, end=time_now, auto_adjust=False)
        if data.empty:
            data = ticker.history(start=start_date, end=yesterday, auto_adjust=False)
        else:    
            data = ticker.history(start=start_date, end=time_now, auto_adjust=False)
        return data
    except Exception as e:
        print(f"Error: {e}")

@st.cache_data(ttl=1800)
def predict_all(tickers, time_now):
    tickers = tickers.keys()

    for ticker in tickers:
        # Fetch data with caching
        data = fetch_data_yfinance(ticker, 60, time_now)
        # Store the fetched data in session_state
        st.session_state['cached_data'][ticker] = data
#predict by data


#session
if 'last_update_time' not in st.session_state:
    st.session_state['last_update_time'] = datetime.now() - timedelta(minutes=30)  

if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}

if 'predict_result' not in st.session_state:
    st.session_state['predict_result'] = {}


#main function
def main():
    #cek jika waktu sudah 30 menit berlalu
    current_time = datetime.now()
    ticker = st.selectbox('Which company do you want to predict?', list(companies.keys()))

    if (current_time - st.session_state['last_update_time']).total_seconds() >= 1800:
        st.session_state['last_update_time'] = current_time
        
        predict_all(companies, current_time)
        if ticker in st.session_state['cached_data']:
            data = st.session_state['cached_data'][ticker]
            st.dataframe(data)
    else:
        time_left = 1800 - (current_time - st.session_state['last_update_time']).total_seconds()
        st.info(f"Next update in {int(time_left / 60)} minutes, last updated at {st.session_state['last_update_time']}")
        
        if ticker in st.session_state['cached_data']:
            data = st.session_state['cached_data'][ticker]
            st.dataframe(data)



if __name__ == "__main__":
    main()