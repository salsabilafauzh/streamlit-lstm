import json
import threading
from scipy import stats
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
import base64
from streamlit_echarts import st_echarts
from pyecharts.globals import ThemeType

from pages.session_config.background_fetcher import fetch_all
from pages.session_config.data_testing_prediction import load_predicted_line_json, save_predicted_line_json
from pages.session_config.history_training import load_training_history, save_training_history
from pages.session_config.lang import get_translation
from pages.session_config.fetched_data_to_json import  save_cached_data
from pages.session_config.fetched_data_to_json import is_over_one_month
from pages.session_config.session_check import session_start
from pages.session_config.store_prediction import load_predictions, save_predictions

session_start()
st.set_page_config(
    page_title="Prediction - Telecommunication",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        body {
        background-color: #FFFFFF;
        }
        #MainMenu {visibility: hidden;}
        .stAppToolbar {visibility: hidden;}
        .stDeployButton {display:none;}
        footer {visibility: hidden;}
        #stDecoration {display:none;}
    </style>
""", unsafe_allow_html=True)

def get_base64_image(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
logo1 = get_base64_image("./images/logo_kampus.png")
logo2 = get_base64_image("./images/logo_kampus_merdeka.png")
logo3 = get_base64_image("./images/logo_kemendikbud.png")
st.markdown(f"""
    <style>
        .logo-container {{
            position: absolute;
            top: 10px;
            right: 20px;
            display: flex;
            gap: 10px;
        }}
        .logo-container img {{
            height: 50px;
        }}
    </style>
    <div class="logo-container">
        <img src="data:image/png;base64,{logo1}" />
        <img src="data:image/png;base64,{logo2}" />
        <img src="data:image/png;base64,{logo3}" />
    </div>
""", unsafe_allow_html=True)



st.markdown(
    r"""
    <style>
    .stAppDeployButton {
            visibility: hidden;
        }
    label[data-testid="stWidgetLabel"] p {
        font-size: 18px !important;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True
)

companies = {
    "TLKM.JK": "Telkom Indonesia Tbk [TLKM]",
    "ISAT.JK": "Indosat Tbk [ISAT]",
    "EXCL.JK": "XL Axiata Tbk [EXCL]"
}
trends = {
    "WEEKLY" :7,
    "BIWEEKLY" : 14,
    "MONTHTLY" : 30
}
language_options = {
    "English": "en-US",
    "Indonesia": "id",
}

features = ["Open","High","Low","Close"]
DATA_DIR = "data"
HISTORY_FILE = "training_history"
CACHE_DATA_FILE = "cached_data.json"

#handling view
def handling_view():
    st.text(f"{get_translation(st.session_state['selected_language'], 'something_wrong')}")
    

def safe_candlestick_row(row):
    if row is None or any(v is None for v in row):
        return [None, None, None, None]
    return row


def plot_data(existing_data, predicted_df, ticker, plot_type, lower_ci=None, upper_ci=None):
    st.subheader(f"{get_translation(st.session_state['selected_language'], 'title_prediction_result')}")

    if predicted_df.index is None or len(predicted_df.index) == 0:
        last_date = existing_data.index[-1]
        next_business_day = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=1)[0]
        predicted_df.index = pd.bdate_range(start=next_business_day, periods=len(predicted_df))

    combined_index = existing_data.index.append(predicted_df.index)
    x_axis_data = combined_index.strftime("%Y-%m-%d").tolist()

    if plot_type == 'candle':
        combined_index = existing_data.index.append(predicted_df.index)
        x_axis_data = combined_index.strftime("%Y-%m-%d").tolist()

        actual_data = existing_data[["Open", "Close", "Low", "High"]].values.tolist()
        predicted_data = predicted_df[["Open", "Close", "Low", "High"]].values.tolist()

        padded_actual = actual_data + [[None, None, None, None]] * len(predicted_data)
        padded_actual = [safe_candlestick_row(r) for r in padded_actual]

        padded_predicted = [[None, None, None, None]] * len(actual_data) + predicted_data
        padded_predicted = [safe_candlestick_row(r) for r in padded_predicted]

        series = [
            {
                "name": get_translation(st.session_state['selected_language'], 'existing_data'),
                "type": "candlestick",
                "data": padded_actual,
                "itemStyle": {
                    "color": "#00da3c",
                    "color0": "#ec0000",
                    "borderColor": "#008F28",
                    "borderColor0": "#8A0000"
                },
            },
            {
                "name": get_translation(st.session_state['selected_language'], 'data_prediction'),
                "type": "candlestick",
                "data": padded_predicted,
                "itemStyle": {
                    "color": "#5B8FF9",
                    "color0": "#4AB9F4",
                    "borderColor": "#5B8FF9",
                    "borderColor0": "#4AC1F4"
                },
            }
        ]

       
        data_test_prediction = load_predicted_line_json(ticker)
        prediction_ohlc_df = data_test_prediction['prediction'].apply(pd.Series)
        if 'predicted_data_testing' not in st.session_state or st.session_state['predicted_data_testing'] is None:
            st.session_state['predicted_data_testing'] = data_test_prediction
        colors = {
            "Open": "#0000FF",   
            "High": "#FFA500",   
            "Low": "#008000",     
            "Close": "#800080"    
        }

        pad_len = len(existing_data) - len(data_test_prediction['date'])
        pad_len = max(0, pad_len)  

        for feature in ["Open", "High", "Low", "Close"]:
            feature_values = prediction_ohlc_df[feature].tolist()
            padded_feature_values = [None]*pad_len + feature_values + [None]* (len(x_axis_data) - pad_len - len(feature_values))
            
            series.append({
                "name": f"Predicted {feature} (data testing)",
                "type": "line",
                "data": padded_feature_values,
                "itemStyle": {"color": colors[feature]},
                "lineStyle": {"color": colors[feature], 
                "width": 1, 
                "type": "dashed"
                },
                "symbol": "none",
            })


        if lower_ci is not None and upper_ci is not None:
            lower_data = [None] * len(existing_data) + list(lower_ci)
            upper_data = [None] * len(existing_data) + list(upper_ci)

            band_data = []
            for u, l in zip(upper_data, lower_data):
                if u is not None and l is not None:
                    band_data.append(u - l)
                else:
                    band_data.append(None)

            
            ci_lower = {
                "name": "95% Confidence Interval - min",
                "type": "line",
                "data": lower_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0},
                "symbol": "none",
                "areaStyle": {"opacity": 0},
                "showLegend": True,
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},
            }

            ci_upper = {
                "name": "95% Confidence Interval - max",
                "type": "line",
                "data": upper_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0}, 
                "symbol": "none",             
                "areaStyle": {"opacity": 0},  
                "showLegend": True,  
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},         
            }

            ci_band = {
                "name": "95% Confidence Interval - difference",
                "type": "line",
                "data": band_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0},
                "symbol": "none",
                "areaStyle": {"color": "rgba(255, 0, 0, 0.5)"},
                "showLegend": True, 
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},
            }

            series.append(ci_lower)
            series.append(ci_band)
            series.append(ci_upper)

            options = {
            "tooltip": {
                "trigger": "axis",
                "axisPointer": {
                    "type": "cross",
                    "crossStyle": {
                        "color": "#FFA500"  
                    }
                }
            },
            "legend": {
                "data": [
                    {
                        "name": get_translation(st.session_state['selected_language'], 'existing_data'),
                        "icon": "roundRect",
                    },
                    {
                        "name": get_translation(st.session_state['selected_language'], 'data_prediction'),
                        "icon": "rect",
                    },
                    {
                        "name": "Predicted Open (data testing)",
                        "icon": "line",
                        "lineStyle": {"color": colors["Open"], "width": 1, "type": "dashed"},
                        "itemStyle": {"color": colors["Open"]},
                    },
                    {
                        "name": "Predicted High (data testing)",
                        "icon": "line",
                        "lineStyle": {"color": colors["High"], "width": 1, "type": "dashed"},
                        "itemStyle": {"color": colors["High"]}
                    },
                    {
                        "name": "Predicted Low (data testing)",
                        "icon": "line",
                        "lineStyle": {"color": colors["Low"], "width": 1, "type": "dashed"},
                        "itemStyle": {"color": colors["Low"]}
                    },
                    {
                        "name": "Predicted Close (data testing)",
                        "icon": "line",
                        "lineStyle": {"color": colors["Close"], "width": 1, "type": "dashed"},
                        "itemStyle": {"color": colors["Close"]}
                    },
                ],
                "top": "top",
                "selectedMode": "multiple",
            },
            "xAxis": {
                "type": "category",
                "data": x_axis_data,
                "scale": True,
                "boundaryGap": False,
                "axisLine": {"onZero": False},
                "splitLine": {"show": False},
            },
            "yAxis": {
                "scale": True,
                "splitArea": {"show": True},
                "min": "dataMin",
                "max": "dataMax",
                "name": "Price",
                "nameLocation": "middle",
                "nameGap": 50,
                "axisLine": {"onZero": False},
            },
            "dataZoom": [
                {"type": "inside", "start": 60, "end": 100},
                {"show": True, "type": "slider", "top": "90%", "start": 60, "end": 100},
            ],
            "series": series,
        }


        st_echarts(options=options, height="650px")

    else:
        colors = {
            "Open": "#1f77b4",    
            "High": "#ff7f0e",    
            "Low": "#2ca02c",   
            "Close": "#c227d6",  
        }

        series = []
        for feature in ["Open", "High", "Low", "Close"]:
            series.append({
                "name": f"Existing {feature}",
                "type": "line",
                "data": [[x, y] for x, y in zip(existing_data.index.strftime("%Y-%m-%d"), existing_data[feature])],
                "lineStyle": {"color": colors[feature]},
                "itemStyle": {"color": colors[feature]}, 
                "showSymbol": False,
            })
            series.append({
                "name": f"Predicted {feature}",
                "type": "line",
                "data": [[x, y] for x, y in zip(predicted_df.index.strftime("%Y-%m-%d"), predicted_df[feature])],
                "lineStyle": {"color": colors[feature], "type": "dashed"},
                "itemStyle": {"color": colors[feature]}, 
                "showSymbol": False,
            })

        if lower_ci is not None and upper_ci is not None:
            lower_data = [None] * len(existing_data) + list(lower_ci)
            upper_data = [None] * len(existing_data) + list(upper_ci)

            band_data = []
            for u, l in zip(upper_data, lower_data):
                if u is not None and l is not None:
                    band_data.append(u - l)
                else:
                    band_data.append(None)

            
            ci_lower = {
                "name": "95% Confidence Interval - min",
                "type": "line",
                "data": lower_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0},
                "symbol": "none",
                "areaStyle": {"opacity": 0},
                "showLegend": True,
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},
            }

            ci_upper = {
                "name": "95% Confidence Interval - max",
                "type": "line",
                "data": upper_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0}, 
                "symbol": "none",             
                "areaStyle": {"opacity": 0},  
                "showLegend": True,   
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},        
            }

            ci_band = {
                "name": "95% Confidence Interval - difference",
                "type": "line",
                "data": band_data,
                "stack": "confidence",
                "lineStyle": {"opacity": 0},
                "symbol": "none",
                "areaStyle": {"color": "rgba(255, 0, 0, 0.2)"},
                "showLegend": True, 
                "itemStyle": {"color": "rgba(255, 0, 0, 0.5)"},
            }

            series.append(ci_lower)
            series.append(ci_band)
            series.append(ci_upper)



        options = {
            "title": {"text": ticker if ticker else "Stock Price Prediction"},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "cross"}},
            "legend": {
                "data": [s["name"] for s in series],
                "top": "top",
            },
            "xAxis": {
                "type": "category",
                "data": x_axis_data,
                "scale": True,
                "boundaryGap": False,
                "axisLine": {"onZero": False},
            },
            "yAxis": {
                "scale": True,
                "splitArea": {"show": True},
                "min": "dataMin",
                "max": "dataMax",
                "name": "Price",
                "nameLocation": "middle",
                "nameGap": 50,
                "axisLine": {"onZero": False},
            },
            "dataZoom": [
                {"type": "inside", "start": 60, "end": 100},
                {"show": True, "type": "slider", "top": "90%", "start": 60, "end": 100},
            ],
            "series": series,
        }

        st_echarts(options=options, height="650px")


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
      st.markdown(
    f"""
    <p style='font-size: 18px; margin-bottom: 15px;'>
        {get_translation(st.session_state['selected_language'], "configuration_title")}
    </p>
    """,
    unsafe_allow_html=True
)

    with col2:
            st.radio(
            get_translation(st.session_state['selected_language'], "change_graph"),
            ["candle", "time series"],
            key="plot_type",
            horizontal=True,
        )
            st.radio(
            get_translation(st.session_state['selected_language'], "change_trend"),
            [ "WEEKLY","BIWEEKLY","MONTHLY"],
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
    elif st.session_state['trend_type'] == 'BIWEEKLY':
        predicted_data = st.session_state['biweekly_prediction'][ticker]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=14, freq="D")
    elif st.session_state['trend_type'] == 'MONTHLY':
        predicted_data = st.session_state['monthly_prediction'][ticker]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=30, freq="D")

    predicted_df = pd.DataFrame(predicted_data, index=future_dates, columns=existing_data.columns)

    data = pd.concat([existing_data, predicted_df])
    desc_sorted_data = data.sort_index(ascending=False)
    
    styled_sorted_df = desc_sorted_data

    lower_ci_key = f"{st.session_state['trend_type'].lower()}_lower_ci"
    upper_ci_key = f"{st.session_state['trend_type'].lower()}_upper_ci"
  
    lower_ci = st.session_state.get(lower_ci_key, {}).get(ticker, None)
    upper_ci = st.session_state.get(upper_ci_key, {}).get(ticker, None)

    plot_data(existing_data,predicted_df,ticker,st.session_state.plot_type, 
            lower_ci,
             upper_ci)
    
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

    styled_sorted_df.index.name = "Date"
    st.dataframe(styled_sorted_df,use_container_width=True)
    history_training = load_training_history(ticker)
    plot_history_training(history_training)
    
    footer = st.empty()
    footer.markdown(
    f"""
    <style>
    .footer {{
    width:100%;
    height:60px;  
    background:#6cf;
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


def reshape_data(data, dates, time_step):
    x, y, y_dates = [], [], []
    for i in range(time_step, len(data)):
        x.append(data[i-time_step:i])
        y.append(data[i])
        y_dates.append(dates[i])
    return np.array(x), np.array(y), np.array(y_dates)


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


def recursive_prediction(steps, test_windowed_x, test_windowed_y, test_data, test_dates,model, ticker, scaler):
    predictions = []
    lower_ci = []
    upper_ci = []

    pred_scaled_on_test = model.predict(test_windowed_x)
    pred_inverse_on_test = scaler.inverse_transform(pred_scaled_on_test)
    actual_inverse = scaler.inverse_transform(test_windowed_y)

    save_predicted_line_json(pred_inverse_on_test,test_dates,ticker)

    actual_close = actual_inverse[:, 3]   
    predicted_close = pred_inverse_on_test[:, 3]
    residuals_close = actual_close - predicted_close
    std_residual = np.std(residuals_close)

    st.session_state['std_residual'][ticker] = std_residual

    input_sequence = test_data[-steps:].reshape(1, steps, test_data.shape[1])

    last_date = test_dates[-1]
    pred_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=30)

    for i in range(30):
        predicted_scaled = model.predict(input_sequence)
        predicted_inverse = scaler.inverse_transform(predicted_scaled)[0]
        predictions.append(predicted_inverse)

        scale = std_residual * np.sqrt(i + 1)
        ci = stats.norm.interval(0.95, loc=predicted_inverse[3], scale=scale)
        lower_ci.append(ci[0])
        upper_ci.append(ci[1])

        predicted_scaled_reshaped = predicted_scaled.reshape(1, 1, test_data.shape[1])
        input_sequence = np.append(input_sequence[:, 1:, :], predicted_scaled_reshaped, axis=1)

        if i == 6:
            st.session_state['weekly_prediction'][ticker] = np.array(predictions[:7]).reshape(7, test_data.shape[1])
            st.session_state['weekly_lower_ci'][ticker] = np.array(lower_ci[:7])
            st.session_state['weekly_upper_ci'][ticker] = np.array(upper_ci[:7])
        elif i == 13:
            st.session_state['biweekly_prediction'][ticker] = np.array(predictions[:14]).reshape(14, test_data.shape[1])
            st.session_state['biweekly_lower_ci'][ticker] = np.array(lower_ci[:14])
            st.session_state['biweekly_upper_ci'][ticker] = np.array(upper_ci[:14])
        elif i == 29:
            st.session_state['monthly_prediction'][ticker] = np.array(predictions).reshape(30, test_data.shape[1])
            st.session_state['monthly_lower_ci'][ticker] = np.array(lower_ci)
            st.session_state['monthly_upper_ci'][ticker] = np.array(upper_ci)

    return pred_dates, np.array(predictions), np.array(lower_ci), np.array(upper_ci)


@st.cache_resource  
def load_model_lstm(ticker):
    file_name = companies[ticker]
    if ticker == "TLKM.JK":
        model=load_model(f"./saved_model/{file_name}.h5",compile=False)
    elif ticker == "EXCL.JK":
        model = load_model(f"./saved_model/{file_name}.h5",compile=False)
    elif ticker == "ISAT.JK":
        model = load_model(f"./saved_model/{file_name}.h5",compile=False)
   
    return model


def predict(ticker, data, last_update_time):
    data_features = data[["Open", "High", "Low", "Close"]].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data_features)
    training_data_len_tlkm = int(len(scaled_data) * 0.8)  
    train_data = scaled_data[:training_data_len_tlkm]
    test_data = scaled_data[training_data_len_tlkm:]
    train_dates = data.index[:training_data_len_tlkm]
    test_dates = data.index[training_data_len_tlkm:]

    train_windowed_data_x,train_windowed_data_y, train_windowed_data_dates = reshape_data(train_data,train_dates,5)
    test_windowed_data_x, test_windowed_data_y, test_windowed_data_date= reshape_data(test_data,test_dates,5)
    if is_over_one_month(datetime.now(), last_update_time) or last_update_time is None:
        update_model(train_windowed_data_x,train_windowed_data_y)
        model_lstm = load_model_lstm(ticker)
    else:
        model_lstm = load_model_lstm(ticker)
    recursive_prediction(5,test_windowed_data_x, test_windowed_data_y, test_data,test_windowed_data_date, model_lstm,ticker,scaler)

def update_model(windowed_data_x,windowed_data_y):

    for _, (ticker, _) in enumerate(companies.items()):
        if ticker not in st.session_state['cached_data']:
            data = fetch_data_yfinance(ticker, datetime.now())
            df_selected_data = data[features]
            st.session_state['cached_data'][ticker] = df_selected_data
        model_lstm = retraining_model(windowed_data_x,windowed_data_y,ticker)

        model_lstm.summary()

        model_dir = "./saved_model"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"{companies.get(ticker)}.h5")
        model_lstm.save(model_path)

def load_content():
    all_data = {}
    for ticker in companies.keys():    
        print(f"Fetching data for {ticker}")
        data = fetch_data_yfinance(ticker, datetime.now())
        if data.isnull().values.any():
            data = data.fillna(method='ffill')
        df_selected_data = data[features]
        st.session_state['cached_data'][ticker] = df_selected_data

        local_path_history = os.path.join(DATA_DIR, f"{HISTORY_FILE}_{ticker}.json")
        if not os.path.exists(local_path_history):
            predict(ticker, st.session_state['cached_data'][ticker], None)
            save_predictions()
        else:
            training_hist = load_training_history(ticker)
            training_hist["date"] = datetime.strptime(training_hist["date"], "%Y-%m-%d %H:%M:%S")
            last_update_time = training_hist["date"]
            predict(ticker, st.session_state['cached_data'][ticker], last_update_time)
            save_predictions()
        df_selected_data["Date"] = df_selected_data.index.strftime("%Y-%m-%d %H:%M:%S")
        all_data[ticker] = df_selected_data
        load_predicted_line_json(ticker)
    save_cached_data(all_data)

def load_content_from_cache(ticker):
    if ticker not in st.session_state['cached_data']:
        local_path_raw_data = os.path.join(DATA_DIR, CACHE_DATA_FILE)
        local_path_history = os.path.join(DATA_DIR, f"{HISTORY_FILE}_{ticker}.json")
        if not os.path.exists(local_path_raw_data) or not os.path.exists(local_path_history):
            load_content()
        with open(local_path_raw_data, "r") as f:
            raw_data = json.load(f)
            training_hist = load_training_history(ticker)
            predictions_data_testing = load_predicted_line_json(ticker)
        
        df = pd.DataFrame(raw_data["cached_data"][ticker])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        training_hist["date"] = datetime.strptime(training_hist["date"], "%Y-%m-%d %H:%M:%S")
        st.session_state['cached_data'][ticker] = df
        st.session_state['predicted_data_testing'] = predictions_data_testing
        if ticker not in st.session_state['weekly_prediction'] or st.session_state['biweekly_prediction'] or st.session_state['monthly_prediction']:
            print("No prediction data found, generating new predictions.")
            load_predictions()

def sync_data():
    while True:
        time.sleep(60)
        if fetch_all():
            load_content()
        else:
            print("Data is up to date")
    
def main():
    st.write(f"# {get_translation(st.session_state['selected_language'], 'title')}")
    selected_company = st.selectbox(get_translation(st.session_state['selected_language'], "select_company"), companies.values())
    selected_ticker = next((key for key, value in companies.items() if value == selected_company), None)
    load_content_from_cache(selected_ticker)
    view_setup(selected_ticker)


if 'sync_thread_started' not in st.session_state:
    threading.Thread(target=sync_data, daemon=True).start()
    st.session_state['sync_thread_started'] = True


if __name__ == "__main__":
    main()