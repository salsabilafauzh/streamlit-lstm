# lang.py
translations = {
    "en-US": {
        "title": "Indonesia telecommunication company prediction",
        "select_company": "Which company do you want to predict?",
        "error_message": "Something went wrong. Please try again later.",
        "trend_info_up": "Indication of upward trend 📈",
        "trend_info_down": "Indication of downward trend 📉",
        "train_loss_title": "Training Loss vs Validation Loss latest training model",
        "copyright": "copyright © Salsabila Fauziah",
        "configuration_title":"Select your configuration type:",
        "change_graph":  "Change graph style:",
        "change_trend":"Change trend to predict:",
        "select_language":"Select Language",
        "second": "seconds",
        "training_time_info": "The training time for the model is:",
        "no_training_history": "No training history for this company.",
        "something_wrong": "Something went wrong, please try again.",
        "title_prediction_result": "Prediction Stock Result",
        "select_feature":"Select the feature you want to display:",
        "last_update_at":"Last updated at:",
        "existing_data":"Actual data",
        "data_prediction":"Prediction data",


        
    },
    "id": {
        "title": "Prediksi perusahaan telekomunikasi Indonesia",
        "select_company": "Perusahaan mana yang ingin Anda prediksi?",
        "error_message": "Terjadi kesalahan. Silakan coba lagi nanti.",
        "trend_info_up": "Indikasi tren hari selanjutnya adalah naik 📈",
        "trend_info_down": "Indikasi tren hari selanjutnya adalah turun 📉",
        "train_loss_title": "Loss Pelatihan vs Loss Validasi model pelatihan terbaru",
        "copyright": "hak cipta © Salsabila Fauziah",
        "configuration_title":"Penyesuaian anda:",
        "change_graph":  "Ubah tampilan grafik:",
        "change_trend":"Ubah rentang prediksi:",
        "select_language":"Pilih bahasa:",
        "second": "detik",
        "training_time_info": "Waktu untuk pelatihan model adalah:",
        "no_training_history": "Tidak ada riwayat pelatihan untuk perusahaan ini.",
        "something_wrong": "Ada yang salah, silakan coba lagi.",
        "title_prediction_result": "Hasil Prediksi Saham",
        "select_feature":"Pilih fitur yang ingin ditampilkan:",
        "last_update_at":"Terakhir diperbarui pada:",
        "existing_data":"Data aktual",
        "data_prediction":"Data prediksi",

        
    }
}


def get_translation(language, key):
    return translations.get(language, translations[language]).get(key, "")


