import streamlit as st
import yfinance as yf
import talib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential,load_model # type: ignore
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsolutePercentageError # type: ignore
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import plotly.graph_objs as go 
import plotly.express as px
import os

from pages.session_config.history_training import load_training_history, save_training_history


st.set_page_config(
    page_title="Prediction - Telecommunication",
    page_icon="chart_with_upwards_trend",
    layout="wide"
)
companies = {
    "TLKM.JK": "Telkom Indonesia (Persero) Tbk [TLKM]",
    "ISAT.JK": "Indosat Tbk [ISAT]",
    "EXCL.JK": "XL Axiata Tbk [EXCL]"
}
trends = {
    "WEEKLY" :7,
    "BI_WEEKLY" : 14,
    "MONTHTLY" : 30
}

features = ["Open","High","Low","Close","Adj Close","Volume"]




#tampilan web
def plot_data(existing_data,predicted_df,ticker, plot_type):
    st.subheader("Prediction Results")
    if plot_type == 'candle':
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=existing_data.index, open=existing_data["Open"], high=existing_data["High"],
            low=existing_data["Low"], close=existing_data["Close"], name="Existing Data"
        ))
        
        fig.add_trace(go.Candlestick(
            x=predicted_df.index, open=predicted_df["Open"], high=predicted_df["High"],
            low=predicted_df["Low"], close=predicted_df["Close"],
            name="Predicted Data", increasing_line_color='rgba(255, 0, 0, 0.5)', decreasing_line_color='rgba(255, 0, 0, 0.5)'
        ))
        st.plotly_chart(fig, use_container_width=True)
    else: 
        features = ["Close", "Open", "High", "Low", "Show All"]
        colors = {"Close": "blue", "Open": "green", "High": "orange", "Low": "purple"}
        selected_feature = st.radio("Select Feature to Show:", features, index=0)

        fig = go.Figure()

        if selected_feature == "Show All":
            for feature in ["Close", "Open", "High", "Low"]:
                fig.add_trace(go.Scatter(
                    x=existing_data.index, y=existing_data[feature],
                    mode="lines", name=f"Existing {feature}",
                    line=dict(color=colors[feature])
                ))
                fig.add_trace(go.Scatter(
                    x=predicted_df.index, y=predicted_df[feature],
                    mode="lines", name=f"Predicted {feature}",
                    line=dict(color=colors[feature], dash="dot")
                ))
        else:
            fig.add_trace(go.Scatter(
                x=existing_data.index, y=existing_data[selected_feature],
                mode="lines", name=f"Existing {selected_feature}",
                line=dict(color=colors[selected_feature])
            ))
            fig.add_trace(go.Scatter(
                x=predicted_df.index, y=predicted_df[selected_feature],
                mode="lines", name=f"Predicted {selected_feature}",
                line=dict(color=colors[selected_feature], dash="dot")
            ))

            fig.add_vrect(
                x0=predicted_df.index[0], x1=predicted_df.index[-1],
                fillcolor="red", opacity=0.1, line_width=0
            )

            fig.update_layout(
                title=f"{ticker} Stock Price Prediction",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend"
            )
        st.plotly_chart(fig, use_container_width=True)
  

def plot_history_training(history):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history.get('loss'), mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history.get('val_loss'), mode='lines', name='Validation Loss'))

    fig.update_layout(
        title="Training Loss vs Validation Loss latest training model!",
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)
 

def view_setup(ticker):
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
  
    existing_data = st.session_state['cached_data'][ticker]
    last_date = existing_data.index[-1] 
    if st.session_state['trend_type'] == 'WEEKLY':
        predicted_data = st.session_state['weekly_prediction'][ticker]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=7, freq="D")
    elif st.session_state['trend_type'] == 'BI_WEEKLY':
        predicted_data = st.session_state['biweekly_prediction'][ticker]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq="D")
    elif st.session_state['trend_type'] == 'MONTHLY':
        predicted_data = st.session_state['monthly_prediction'][ticker]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")

    predicted_df = pd.DataFrame(predicted_data, index=future_dates, columns=existing_data.columns)


    data = pd.concat([existing_data, predicted_df])
   
  
    plot_data(existing_data,predicted_df,ticker,st.session_state.plot_type)
    st.dataframe(data,use_container_width=True)
    
    history_training = load_training_history(ticker)
    plot_history_training(history_training)

    st.session_state['isDisable_selector'] = False   


#PRE-PROCESSING
@st.cache_data(ttl=1800)
def fetch_data_yfinance(ticker_company, time_now):
    try:
        ticker = yf.Ticker(ticker_company)
        start_date = time_now - timedelta(days=5*365)
        yesterday = time_now - timedelta(days=1)
        data = ticker.history(start=start_date, end=time_now, auto_adjust=False)
        if data.empty:
            data = ticker.history(start=start_date, end=yesterday, auto_adjust=False)
            return data
        else:
            return data
    except Exception as e:
        print(f"Error: {e}")


def reshape_data(data, time_step):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, :])
        y.append(data[i, :])
    return np.array(X), np.array(y)


#TRAINING & LOAD PREDICTION
@st.cache_resource
def retraining_model(X_train, y_train,ticker):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units = 32, return_sequences = False))
    model.add(Dense(len(features)))
    model.compile(optimizer = "adam",loss = "mean_absolute_error", metrics=[
        RootMeanSquaredError(name='rmse'),
        MeanAbsolutePercentageError(name='mape')])
   
    history = model.fit(X_train, y_train, epochs=300, batch_size=32, validation_split=0.2,verbose=1)
    save_training_history(history.history['loss'], history.history['val_loss'],ticker)
    plot_history_training(history)

    return model


def recursive_prediction(steps, input_data, model,ticker,scaler):
    predictions = []
    input_data = input_data
    input_sequence = input_data[-steps:].reshape(1,5,6)
    
    for i in range(30):
        predicted_scaled = model.predict(input_sequence)

        new_input = np.append(input_sequence[:, 1:, :], predicted_scaled.reshape(1,1,6), axis=1)
        input_sequence = new_input 

        normal_prediction_result = scaler.inverse_transform(predicted_scaled)
        predictions.append(normal_prediction_result)
       
        
        result = np.array(predictions)
        if i == 6:
            st.session_state['weekly_prediction'][ticker] = result.reshape(7,6)
        elif i == 13:
            st.session_state['biweekly_prediction'][ticker] = result.reshape(14,6)
        elif i == 29:
            st.session_state['monthly_prediction'][ticker] = result.reshape(30,6)
        


@st.cache_resource  
def load_model_lstm(ticker):
    file_name = companies.get(ticker)
    if ticker == "TLKM.JK":
        model=load_model(f"./pages/saved_model/{file_name}.h5",compile=False)
    elif ticker == "EXCL.JK":
        model = load_model(f"./pages/saved_model/{file_name}.h5",compile=False)
    elif ticker == "ISAT.JK":
        model = load_model(f"./pages/saved_model/{file_name}.h5",compile=False)
   
    return model

@st.cache_data(ttl=1800)
def predict(ticker, data, current_time):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    windowed_data_x,windowed_data_y = reshape_data(scaled_data,5)
    if current_time >= st.session_state['last_update_time']['next_date_update']:
        model_lstm = retraining_model(windowed_data_x,windowed_data_y,ticker)
        model_lstm.summary()

        model_dir = "./pages/saved_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{companies.get(ticker)}.h5")
        model_lstm.save(model_path)

        st.session_state['last_update_time']['date_fetched'] = current_time
        st.session_state['last_update_time']['next_date_update'] = current_time + relativedelta(months=1)
    else:
        model_lstm = load_model_lstm(ticker)
        model_lstm.summary()
    recursive_prediction(5,scaled_data,model_lstm,ticker,scaler)


def load_content():
    current_time = datetime.now()
    if (current_time - st.session_state['last_update_time']['time_yfinance_fetched']).total_seconds() >= 1800:
        tickers = companies.keys()
        for ticker in tickers:
            data = fetch_data_yfinance(ticker, current_time)
            df_selected_data = data[features]
            st.session_state['cached_data'][ticker] = df_selected_data

        st.session_state['last_update_time']['time_yfinance_fetched'] = current_time
    else:
        time_left = 1800 - (current_time - st.session_state['last_update_time']['time_yfinance_fetched']).total_seconds()
        st.info(f"Next update in {int(time_left / 60)} minutes, last updated at {st.session_state['last_update_time']['time_yfinance_fetched']}")

def main():
    st.write("# Indonesia telecommunication company prediction!")

    selected_company = st.selectbox('Which company do you want to predict?', companies.values())
    selected_ticker = next((key for key, value in companies.items() if value == selected_company), None)

    load_content()

    if selected_ticker not in st.session_state.get('predicted_data', {}):
        predict(selected_ticker, st.session_state['cached_data'][selected_ticker].values, datetime.now())

    view_setup(selected_ticker)


 #konfigurasi session
if 'last_update_time' not in st.session_state:
    st.session_state['last_update_time'] = {}
    st.session_state['last_update_time']['date_fetched'] = datetime.now()
    st.session_state['last_update_time']['next_date_update'] = datetime.now() + relativedelta(months=1)
    st.session_state['last_update_time']['time_yfinance_fetched'] = datetime.now() - timedelta(minutes=30)  

if 'cached_data' not in st.session_state:
    st.session_state['cached_data'] = {}

if 'predict_result' not in st.session_state:
    st.session_state['predict_result']={}

if 'weekly_prediction' not in st.session_state:
    st.session_state['weekly_prediction']={}

if 'biweekly_prediction' not in st.session_state:
    st.session_state['biweekly_prediction']={}

if 'monthly_prediction' not in st.session_state:
    st.session_state['monthly_prediction']={}

if 'plot_type' not in st.session_state:
    st.session_state.plot_type = "time series"

if 'trend_type' not in st.session_state:
    st.session_state.trend_type = "WEEKLY"

if 'isDisable_selector' not in st.session_state:
    st.session_state.isDisable_selector = True


if __name__ == "__main__":
    main()