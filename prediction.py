import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM,Dense, Dropout # type: ignore
from tensorflow.keras.models import Sequential,load_model # type: ignore
from dateutil.relativedelta import relativedelta
from datetime import datetime, timedelta
import plotly.graph_objs as go
import os
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import time

from pages.session_config.history_training import load_training_history, save_training_history
from pages.session_config.lang import get_translation 

st.set_page_config(
    page_title="Prediction - Telecommunication",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        .reportview-container {
            margin-top: -2em;
        }
        #MainMenu {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)


st.markdown(
    r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    </style>
    """, unsafe_allow_html=True
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
language_options = {
    "English": "en-US",
    "Indonesia": "id",
}

features = ["Open","High","Low","Close"]

#handling view
def handling_view():
    st.text(f"{get_translation(st.session_state['selected_language'], 'something_wrong')}")

#tampilan web
def plot_data(existing_data,predicted_df,ticker, plot_type):
    st.subheader(f"{get_translation(st.session_state['selected_language'], 'title_prediction_result')}")
    if plot_type == 'candle':
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=existing_data.index, open=existing_data["Open"], high=existing_data["High"],
            low=existing_data["Low"], close=existing_data["Close"], name=f"{get_translation(st.session_state['selected_language'], 'existing_data')}",
        ))
        
        fig.add_trace(go.Candlestick(
            x=predicted_df.index, open=predicted_df["Open"], high=predicted_df["High"],
            low=predicted_df["Low"], close=predicted_df["Close"],
            name=f"{get_translation(st.session_state['selected_language'], 'data_prediction')}", increasing_line_color='rgba(30, 144, 255, 0.5)', decreasing_line_color='rgba(138, 43, 226, 0.5)'
        ))
        fig.update_layout(
                title=f"{ticker}",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend",
                dragmode="pan"
            )
        st.plotly_chart(fig, use_container_width=True)
    else: 
        features = ["Close", "Open", "High", "Low", "Show All"]
        colors = {"Close": "blue", "Open": "green", "High": "orange", "Low": "purple"}
        selected_feature = st.radio(f"{get_translation(st.session_state['selected_language'], 'select_feature')}", features, index=0)

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
                fig.update_layout(
                title=f"{ticker}",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend",
                dragmode="pan"
            )
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
                title=f"{ticker}",
                xaxis_title="Date",
                yaxis_title="Stock Price",
                legend_title="Legend",
                dragmode="pan"
            )
        st.plotly_chart(fig, use_container_width=True)

def plot_history_training(history):
    st.subheader(f"{get_translation(st.session_state['selected_language'], 'training_time_info')}: {history['training_time']} {get_translation(st.session_state['selected_language'], 'second')}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=history['loss'], mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=history['val_loss'], mode='lines', name='Validation Loss'))

    fig.update_layout(
        title=get_translation(st.session_state['selected_language'], "train_loss_title"),
        xaxis_title="Epochs",
        yaxis_title="Loss",
        legend=dict(x=0, y=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def highlight_predicted_data(row):
    if isinstance(row.name, pd.Timestamp): 
        index_value = row.name.toordinal()  
    else:
        index_value = row.name  
    num_days = trends.get(st.session_state["trend_type"], 7)

    num_cols = len(row)  

    if index_value < num_days:
        return ["background-color: red"] * num_cols 
    return [""] * num_cols  

def change_language():
    lang_code = language_options[st.session_state["language_selector"]]
    st.session_state["selected_language"] = lang_code
    st.session_state["language_selector"] = next((k for k, v in language_options.items() if v == lang_code), None)

def lock_ui():
    st.markdown("""
    <style>
    body, .block-container, .sidebar, .stButton > button {
        pointer-events: none !important;
        opacity: 0.5 !important;
        cursor: wait !important;
    }
    </style>
    """, unsafe_allow_html=True)

def unlock_ui():
    st.markdown("""
    <style>
    body, .block-container, .sidebar, .stButton > button {
        pointer-events: auto !important;
        opacity: 1 !important;
        cursor: default !important;
    }
    </style>
    """, unsafe_allow_html=True)


def view_setup(ticker):
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(get_translation(st.session_state['selected_language'], "configuration_title"))
    with col2:
            st.radio(
            get_translation(st.session_state['selected_language'], "change_graph"),
            ["candle", "time series"],
            key="plot_type",
            horizontal=True,
        )
            st.radio(
            get_translation(st.session_state['selected_language'], "change_trend"),
            [ "WEEKLY","BI_WEEKLY","MONTHLY"],
            key="trend_type",
            horizontal=True,
        )
    with col3:
        selected_lang = st.radio(get_translation(st.session_state['selected_language'], "select_language"), list(language_options.keys()),  key="language_selector",
        horizontal=True,
        on_change=change_language)
        st.markdown(f"**{get_translation(st.session_state['selected_language'], 'selected_language')}**: {selected_lang} (`{selected_lang}`)")
   
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
    desc_sorted_data = data.sort_index(ascending=False)
    
    styled_sorted_df = desc_sorted_data.style.apply(highlight_predicted_data, axis=1)

    plot_data(existing_data,predicted_df,ticker,st.session_state.plot_type)
    
    existing_avg = existing_data['Close'].tail(1).values[0]
    predicted_avg = predicted_df['Close'].mean() 

    if existing_avg > predicted_avg:
        trend_info = get_translation(st.session_state['selected_language'], "trend_info_down")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: auto; background-color: #E74C3C; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
                <h3 style=""> {trend_info}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        trend_info = get_translation(st.session_state['selected_language'], "trend_info_up")
        st.markdown(
            f"""
            <div style="display: flex; justify-content: center; align-items: center; height: auto; background-color: #2ECC71; border-radius: 5px; padding: 10px; margin-bottom: 20px;">
                <h3 style=""> {trend_info}</h3>
            </div>
            """,
            unsafe_allow_html=True
    )


    st.dataframe(styled_sorted_df,use_container_width=True)
    history_training = load_training_history(ticker)
    if history_training is None:
        st.warning(get_translation(st.session_state['selected_language'], "no_training_history"))
    else:
        plot_history_training(history_training)

    st.markdown(
    f"""
    <style>
    .footer {{
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #666;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #ccc;
    }}
    </style>
    <div class="footer">
        <b>{get_translation(st.session_state['selected_language'], 'copyright')}</b>
    </div>
    """,
    unsafe_allow_html=True
)


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
    lock_ui()
    if ticker == "TLKM.JK" or "EXCL.JK":
        model = Sequential()
        model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 64))
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.compile(optimizer = "adam",loss = "mean_absolute_error", metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape')])
    elif ticker == "ISAT.JK":
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(0.2))
        model.add(LSTM(units = 50))
        model.add(Dropout(0.2))
        model.add(Dense(4))
        model.compile(optimizer = "adam",loss = "mean_absolute_error", metrics=[
            tf.keras.metrics.RootMeanSquaredError(name='rmse'),
            tf.keras.metrics.MeanAbsolutePercentageError(name='mape')])

    y_train = y_train[:, :4]

    start_time = time.time()

    if ticker == "TLKM.JK":
        epochs = 300
    else:
        epochs = 200
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=1, validation_split=0.2)
    end_time = time.time()
    training_time = end_time - start_time

    save_training_history(history.history['loss'], history.history['val_loss'],training_time,ticker)
    unlock_ui()
    return model


def recursive_prediction(steps, input_data, model,ticker,scaler):
    predictions = []
    input_sequence = input_data[-steps:].reshape(1,steps,4)
    
    for i in range(30):
        predicted_scaled = model.predict(input_sequence)
        
        new_input = np.append(input_sequence[:, 1:, :], predicted_scaled.reshape(1,1,4), axis=1)
        input_sequence = new_input 

        normal_prediction_result = scaler.inverse_transform(predicted_scaled)
        predictions.append(normal_prediction_result)
       
        result = np.array(predictions).astype(int)
        if i == 6:
            st.session_state['weekly_prediction'][ticker] = result.reshape(7,4)
        elif i == 13:
            st.session_state['biweekly_prediction'][ticker] = result.reshape(14,4)
        elif i == 29:
            st.session_state['monthly_prediction'][ticker] = result.reshape(30,4)
        


@st.cache_resource  
def load_model_lstm(ticker):
    file_name = companies[ticker]
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
        update_model(windowed_data_x,windowed_data_y,current_time)
        model_lstm = load_model_lstm(ticker)
    else:
        model_lstm = load_model_lstm(ticker)
        model_lstm.summary()
    recursive_prediction(5,scaled_data,model_lstm,ticker,scaler)


def update_model(windowed_data_x,windowed_data_y,current_time):

    for _, (ticker, _) in enumerate(companies.items()):
        if ticker not in st.session_state['cached_data']:
            data = fetch_data_yfinance(ticker, datetime.now())
            df_selected_data = data[features]
            st.session_state['cached_data'][ticker] = df_selected_data
        model_lstm = retraining_model(windowed_data_x,windowed_data_y,ticker)

        model_lstm.summary()

        model_dir = "./pages/saved_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{companies.get(ticker)}.h5")
        model_lstm.save(model_path)

        st.session_state['last_update_time']['date_fetched'] = current_time
        st.session_state['last_update_time']['next_date_update'] = current_time + relativedelta(months=1)

def load_content():
    current_time = datetime.now()
    is_need_update =  is_need_update_data()
    if is_need_update :
        tickers = companies.keys()
        for ticker in tickers:
            data = fetch_data_yfinance(ticker, current_time)
            df_selected_data = data[features]
            st.session_state['cached_data'][ticker] = df_selected_data
            st.session_state['last_update_time']['time_yfinance_fetched'] = current_time
            predict(ticker, st.session_state['cached_data'][ticker], datetime.now())
    else:
        # time_left = 1800 - (current_time - st.session_state['last_update_time']['time_yfinance_fetched']).total_seconds()
        translated_text = get_translation(st.session_state['selected_language'], "last_update_at")
        st.info(f"{translated_text} {st.session_state['last_update_time']['time_yfinance_fetched']}")
    


def is_need_update_data():
    current_time = datetime.now()
    new_data = fetch_data_yfinance("TLKM.JK", current_time)
    cached_df = st.session_state['cached_data'].get("TLKM.JK")

    if cached_df is None:
        return True

    if len(new_data) != len(cached_df):
        return True

    if new_data.index[-1] != cached_df.index[-1]:
        return True

    return False
    
def main():
    # try:
        st.write(f"# {get_translation(st.session_state['selected_language'], 'title')}")
        selected_company = st.selectbox(get_translation(st.session_state['selected_language'], "select_company"), companies.values())
        selected_ticker = next((key for key, value in companies.items() if value == selected_company), None)
        
        load_content()
        view_setup(selected_ticker)
    # except:
    #     handling_view()



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


if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = "id"

if "language_selector" not in st.session_state:
    st.session_state["language_selector"] = "Indonesia"


if __name__ == "__main__":
    main()