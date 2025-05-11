import os
import json
import streamlit as st
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
TICKERS = ["TLKM.JK", "ISAT.JK", "EXCL.JK"]
import json

def save_predictions():
    saved_data = {
        "weekly_prediction": {},
        "biweekly_prediction": {},
        "monthly_prediction": {}
    }

    for period in ["weekly_prediction", "biweekly_prediction", "monthly_prediction"]:
        for ticker, arr in st.session_state.get(period, {}).items():
            saved_data[period][ticker] = arr.tolist()

    with open(os.path.join(DATA_DIR, "predictions.json"), "w") as f:
        json.dump(saved_data, f)



def load_predictions():
    path = os.path.join(DATA_DIR, "predictions.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            saved_data = json.load(f)
            for period in ["weekly_prediction", "biweekly_prediction", "monthly_prediction"]:
                st.session_state[period] = {
                    ticker: np.array(data) for ticker, data in saved_data.get(period, {}).items()
                }