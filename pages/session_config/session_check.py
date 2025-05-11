
import streamlit as st
def session_start():

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