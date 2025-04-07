# lang.py

translations = {
    "en": {
        "title": "Indonesia telecommunication company prediction.",
        "select_company": "Which company do you want to predict?",
        "error_message": "Something went wrong. Please try again later.",
        "trend_info_up": "Indication of upward trend 📈",
        "trend_info_down": "Indication of downward trend 📉",
        "train_loss_title": "Training Loss vs Validation Loss latest training model",
        "copyright": "copyright © Salsabila Fauziah",
        "configuration_title":"Select your configuration type:",
        "change_graph":  "Change graph style:",
        "change_trend":"Change trend to predict:",
        "select_language":"Select Language"
    },
    "id": {
        "title": "Prediksi perusahaan telekomunikasi Indonesia.",
        "select_company": "Perusahaan mana yang ingin Anda prediksi?",
        "error_message": "Terjadi kesalahan. Silakan coba lagi nanti.",
        "trend_info_up": "Indikasi tren naik 📈",
        "trend_info_down": "Indikasi tren turun 📉",
        "train_loss_title": "Loss Pelatihan vs Loss Validasi model pelatihan terbaru",
        "copyright": "hak cipta © Salsabila Fauziah",
        "configuration_title":"Penyesuaian anda:",
        "change_graph":  "Ubah tampilan grafik:",
        "change_trend":"Ubah rentang prediksi:",
        "select_language":"Pilih bahasa:"
    }
}

def get_translation(language, key):
    return translations.get(language, translations['en']).get(key, "")
