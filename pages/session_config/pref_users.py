import streamlit as st
import os
import toml

language_options = {
    "English": "en",
    "Indonesian": "id",
}

CONFIG_PATH = os.path.expanduser("~/.streamlit/config.toml")


def update_config(lang):
    config_data = {
        "global": {
            "language": lang
        },
        "theme": {
            "primaryColor": "#ff4b4b",
            "backgroundColor": "#f4f4f4",
            "textColor": "#262730"
        }
    }
    with open(CONFIG_PATH, "w") as config_file:
        toml.dump(config_data, config_file)
translations = {
    "en": {
        "page_title": "Prediction - Telecommunication",
        "select_configuration": "Select your configuration type:",
        "change_graph_style": "Change graph style:",
        "change_trend": "Change trend to predict:",
        "select_language": "Select Language",
        "selected_language": "Selected Language",
        "prediction_results": "Prediction Results",
        "select_feature": "Select Feature to Show:",
        "train_loss": "Train Loss",
        "val_loss": "Validation Loss",
        "training_loss_vs_validation": "Training Loss vs Validation Loss latest training model",
        "epochs": "Epochs",
        "loss": "Loss",
        "stock_price_prediction": "{} Stock Price Prediction",
        "date": "Date",
        "stock_price": "Stock Price",
        "legend": "Legend",
        "trend_down": "Indicates a downward trend",
        "trend_up": "Indicates an upward trend",
        "copyright": "copyright © salsabila fauziah",
    },
    "id": {
        "page_title": "Prediksi - Telekomunikasi",
        "select_configuration": "Pilih jenis konfigurasi Anda:",
        "change_graph_style": "Ubah gaya grafik:",
        "change_trend": "Ubah tren yang diprediksi:",
        "select_language": "Pilih Bahasa",
        "selected_language": "Bahasa yang Dipilih",
        "prediction_results": "Hasil Prediksi",
        "select_feature": "Pilih Fitur yang Ditampilkan:",
        "train_loss": "Kerugian Pelatihan",
        "val_loss": "Kerugian Validasi",
        "training_loss_vs_validation": "Kerugian Pelatihan vs Kerugian Validasi dari model terbaru",
        "epochs": "Epoch",
        "loss": "Kerugian",
        "stock_price_prediction": "Prediksi Harga Saham {}",
        "date": "Tanggal",
        "stock_price": "Harga Saham",
        "legend": "Legenda",
        "trend_down": "Indikasi tren bergerak turun",
        "trend_up": "Indikasi tren bergerak naik",
        "copyright": "hak cipta © salsabila fauziah",
    }
}


if "selected_language" not in st.session_state:
    st.session_state["selected_language"] = "en"

selected_lang = st.radio("Select Language", list(language_options.keys()), horizontal=True)
lang_code = language_options[selected_lang]

if lang_code != st.session_state["selected_language"]:
    st.session_state["selected_language"] = lang_code
    update_config(lang_code)  
    st.rerun()

st.markdown(f"**Selected Language:** {selected_lang} (`{lang_code}`)")