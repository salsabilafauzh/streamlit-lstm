import os
import json
import streamlit as st
import numpy as np

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
TICKERS = ["TLKM.JK", "ISAT.JK", "EXCL.JK"]

def save_predictions():
    saved_data = {
        "weekly_prediction": {},
        "weekly_lower_ci": {},
        "weekly_upper_ci": {},
        "biweekly_prediction": {},
        "biweekly_lower_ci": {},
        "biweekly_upper_ci": {},
        "monthly_prediction": {},
        "monthly_lower_ci": {},
        "monthly_upper_ci": {},
        "std_residual": {}
    }

    for period in ["weekly", "biweekly", "monthly"]:
        for ticker, arr in st.session_state.get(f"{period}_prediction", {}).items():
            saved_data[f"{period}_prediction"][ticker] = arr.tolist()
        
        for ticker, arr in st.session_state.get(f"{period}_lower_ci", {}).items():
            saved_data[f"{period}_lower_ci"][ticker] = arr.tolist()
        
        for ticker, arr in st.session_state.get(f"{period}_upper_ci", {}).items():
            saved_data[f"{period}_upper_ci"][ticker] = arr.tolist()

    for ticker, val in st.session_state.get("std_residual", {}).items():
        saved_data["std_residual"][ticker] = float(val)

    path = os.path.join(DATA_DIR, "predictions.json")
    with open(path, "w") as f:
        json.dump(saved_data, f)

def load_predictions():
    path = os.path.join(DATA_DIR, "predictions.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            saved_data = json.load(f)

        for period in ["weekly", "biweekly", "monthly"]:
            st.session_state[f"{period}_prediction"] = {
                ticker: np.array(data) for ticker, data in saved_data.get(f"{period}_prediction", {}).items()
            }

            st.session_state[f"{period}_lower_ci"] = {
                ticker: np.array(data) for ticker, data in saved_data.get(f"{period}_lower_ci", {}).items()
            }

            st.session_state[f"{period}_upper_ci"] = {
                ticker: np.array(data) for ticker, data in saved_data.get(f"{period}_upper_ci", {}).items()
            }

        st.session_state["std_residual"] = {
            ticker: float(val) for ticker, val in saved_data.get("std_residual", {}).items()
        }

