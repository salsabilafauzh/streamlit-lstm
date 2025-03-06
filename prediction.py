import streamlit as st
import yfinance as yf
import tensorflow as tf
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers.recurrent import LSTM
from tensorflow.python.keras.layers import Dense,  Dropout
from tensorflow.python.keras.models import Sequential, load_model
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from datetime import datetime, timedelta
import plotly.graph_objs as go 
import plotly.express as px

st.set_page_config(
    page_title="Prediction - Telecommunication",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)

st.write("# Indonesia telecomunication company prediction!")
companies = {
    "TLKM.JK": "PT Telkom Indonesia (Persero) Tbk",
    "ISAT.JK": "Indosat Tbk PT",
    "EXCL.JK": "XL Axiata Tbk PT"
}

trend = {
    "WEEKLY" :7,
    "BI_WEEKLY" : 14,
    "MONTHTLY" : 30
}


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
            return data
        else:
            return data
    except Exception as e:
        print(f"Error: {e}")


# @st.cache_data(ttl=1800)
# def retraining_model():
#     model = Sequential()
#     model.add(LSTM(units = 50,return_sequences = True, input_shape = (X_train.shape[1],num_features)))
#     model.add(Dropout(0.2))
#     model.add(LSTM(units = 50, return_sequences = False))
#     model.add(Dropout(0.2))
#     model.add(Dense(num_features))
#     model.compile(optimizer = "adam",loss = "mean_absolute_error")

#    # Training
#     history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1)

#     model.summary()

#     # Buat nama file unik berdasarkan waktu
#     model_path = f"lstm_model_{name_file}.h5"

#     model.save(model_path)
#     return ""


def recursive_prediction(steps,trend, input_data, model):
    predictions = []
    for _ in range(trend):
        # input_sequence = input_data[-steps:].reshape(1, steps, input_data.shape[1])
        input_sequence = input_data[-steps:]
        predicted_scaled = model.predict(input_sequence)
        predictions.append(predicted_scaled)
        
        # Geser window dengan menambahkan prediksi terbaru
        new_input = np.append(input_sequence[:, 1:, :], predicted_scaled.reshape(1, 1, 1), axis=1)
        input_sequence = new_input 
    return predictions

def reshape_data(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)


def load_saved_model():
    model=load_model('./pages/saved_model/best_model.h5')
    return model

def plot_data(data,ticker, plot_type):
    st.subheader("Prediction Results")
    if plot_type == 'candle':
            candlestick_chart = go.Figure(data=[go.Candlestick(x=data.index, open=data['Open'], high=data['High'], low=data['Low'], close=data['Close'])])
            candlestick_chart.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=False)
            st.plotly_chart(candlestick_chart, use_container_width=True)
    else: 
        fig = px.line(data, x=data.index, y='Close', title=f"{ticker} Time Series")
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        st.plotly_chart(fig, use_container_width=True)
  

@st.cache_data(ttl=1800)
def predict(ticker,data):
    features = ["Open","High","Low","Close","Adj Close","Volume"]
    selected_data = data[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(selected_data)
    # windowed_data_x,windowed_data_y = reshape_data(scaled_data,5)

    # model_lstm = load_saved_model()
    # model_lstm.summary()

    # trend_type = trend.keys()
    # predicted_scaled = recursive_prediction(5,trend_type["WEEKLY"],windowed_data_x,model_lstm)
    
    # result_predicted_price = scaler.inverse_transform(predicted_scaled.reshape(-1, 1))

#     for i in range(len(features)):
#         plt.figure(figsize=(9, 6))
#         plt.plot(selected_data[:, i], label=f"Actual - {features[i]}", color='blue')
#         plt.plot(result_predicted_price[:, i], label=f"Predicted - {features[i]}", color='orange')

#         # Add title and labels
#         plt.title(f"Actual vs Predicted for {features[i]}")
#         plt.xlabel("Samples")
#         plt.ylabel("Values")
#         plt.legend()
#         plt.grid()

#         plt.tight_layout()
#         plt.show()
    st.session_state['cached_data'][ticker] = data
    # if 'predict_result' not in st.session_state:
    #     st.session_state['predict_result'][ticker] = result_predicted_price






def main():
    selected_company = st.selectbox('Which company do you want to predict?', companies.values())
    selected_ticker = next((key for key, value in companies.items() if value == selected_company), None)
    current_time = datetime.now()        
    if (current_time - st.session_state['last_update_time']).total_seconds() >= 1800:
        tickers = companies.keys()
        for ticker in tickers:
            data = fetch_data_yfinance(ticker, 180, current_time)
            predict(ticker, data)
            st.session_state['cached_data'][ticker] = data
        st.session_state['last_update_time'] = current_time
    else:
        time_left = 1800 - (current_time - st.session_state['last_update_time']).total_seconds()
        st.info(f"Next update in {int(time_left / 60)} minutes, last updated at {st.session_state['last_update_time']}")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('Select your configuration type:')
    with col2:
            st.radio(
            "Change graph style👇",
            ["candle", "time series"],
            key="plot_type",
            horizontal=True,
        )
            st.radio(
            "Change trend to predict👇",
            [ "WEEKLY","BI_WEEKLY","MONTHLY"],
            key="trend_type",
            horizontal=True,
        )
    plot_data(st.session_state['cached_data'][selected_ticker],selected_ticker,st.session_state.plot_type)
    st.dataframe(st.session_state['cached_data'][selected_ticker])

#konfigurasi session
if 'last_update_time' not in st.session_state:
    st.session_state['last_update_time'] = datetime.now() - timedelta(minutes=30)  

if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}

if 'predict_result' not in st.session_state:
    st.session_state['predict_result'] = {}

if 'plot_type' not in st.session_state:
    st.session_state.plot_type = "time series"

if 'trend_type' not in st.session_state:
    st.session_state.trend_type = "WEEKLY"


if __name__ == "__main__":
    main()