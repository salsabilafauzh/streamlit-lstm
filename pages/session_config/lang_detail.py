def get_translation(language, text_key):
    translations = {
        'en-US': {
            'page_title': "Detail of Implementation",
            'about_this_page_label':"About this page:",
            'about_this_page': "This page provides a comprehensive explanation of the stock prediction implementation, including the techniques used (such as LSTM), data preparation, evaluation metrics, and visualization of both historical and predicted values. Users can explore different features of the stock and understand the prediction results clearly.",
            'tech_stack': "Tech Stack Used to Develop",
            'time_range': "Time Range of Data Used",
            'data_used': "Data used and how to get:",
            'companies_available': "Companies Available:",
            'train_test_split': "Train/Test Split",
            'train_test_split_description': "The 80:20 data split ratio is important to ensure that the machine learning model can be trained using most of the historical data and evaluated using unseen data to effectively measure its prediction accuracy.",
            'lstm_flow': "LSTM Flow Architecture",
            'lstm_cell': "LSTM Cell Architecture",
            'loss_curve': "Loss Curve Interpretation",
            'rmse': "Root Mean Squared Error (RMSE)",
            'mape': "Mean Absolute Percentage Error (MAPE)",
            'copyright': "© 2025 by Salsabila Fauziah",
            "data_used_description": "The data used in this project is sourced from Yahoo Finance. The dataset includes historical stock prices and trading volumes for the selected companies.",
            "time_used":"august 2019 – August 2024",
            "companies_label":"🏢Companies Available:",
            "companies_available_description": "The companies available for stock prediction in this project include PT Telkom Indonesia (Persero) Tbk, Indosat Tbk PT, and XL Axiata Tbk PT. These companies were selected due to their significance and representation within the Indonesian telecommunications sector.",
            "data_test_label":"Test Data",
            "data_train_label":"Train Data",
            "lstm_detail_label":"Description of Long Short Term Memory Implementation",
            "how_lstm_works_label":"How does LSTM works:",
            "lstm_description":"Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture that is capable of learning long-term dependencies. It is particularly effective for time series prediction tasks, such as stock price forecasting. LSTM networks are designed to remember information for long periods of time, making them suitable for sequential data analysis.",
            "inside_of_lstm_label":"Inside a Cell of LSTM:",
            "inside_cell_lstm_description":"""LSTM cells contain three gates: the input gate, forget gate, and output gate. These gates control the flow of information into and out of the cell, allowing it to retain or discard information based on its relevance to the current prediction task.
    The input gate determines how much of the new information should be stored, the forget gate decides what information to discard, and the output gate controls what information is sent to the next layer.
    This gating mechanism enables LSTM networks to learn complex patterns in sequential data and make accurate predictions.
    By using LSTM, we can effectively model the temporal dependencies in stock price movements and improve the accuracy of our predictions.
    The LSTM model is trained on historical stock data, learning to recognize patterns and trends that can inform future price movements.""",
    "loss_curve_label": "Loss Curve Interpretation",
    "loss_curve_description": """During model training, we track both **training loss** and **validation loss**. A good indicator of model performance is when both loss curves gradually converge and stay close. This indicates the model is learning well and is not suffering from underfitting or overfitting.
    """,
    "loss_curve_highlight":"The closer the training and validation loss lines, the more stable and accurate the model becomes.",
    "model_eval_label":"Model Performance Evaluation",
    "model_eval_description":"To evaluate the predictive performance of the Long Short-Term Memory (LSTM) model, we use two common metrics: **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**. These metrics help determine how well the model's predictions align with actual stock prices.",
    "rmse_description":"RMSE measures the average magnitude of the error between the predicted values and the actual values. It penalizes large errors more severely than smaller ones.",
    "rmse_description_2":"A lower RMSE indicates better model performance, as it means the predicted values are closer to the actual values on average.",
    "mape_description":"MAPE measures the average percentage difference between actual and predicted values. It gives a clearer sense of prediction accuracy in percentage terms.",
    "mape_description_2":"A lower MAPE suggests that the model is performing well in minimizing prediction error relative to actual stock prices.",
        },
        'id': {
            'page_title': "Detail Implementasi",
            'about_this_page_label':"Tentang Halaman:",
            'about_this_page': "Halaman ini memberikan penjelasan lengkap mengenai implementasi prediksi saham, termasuk teknik yang digunakan (seperti LSTM), persiapan data, metrik evaluasi, dan visualisasi nilai historis dan prediksi. Pengguna dapat menjelajahi berbagai fitur saham dan memahami hasil prediksi dengan jelas.",
            'tech_stack': "Teknologi yang Digunakan untuk Pengembangan",
            'time_range': "Rentang Waktu Data yang Digunakan",
            'data_used': "Data yang digunakan dan cara mendapatkannya:",
            'companies_available': "Perusahaan yang Tersedia:",
            'train_test_split': "Pembagian Data Latih/Uji",
            'train_test_split_description': "Rasio pembagian data 80:20 ini penting untuk memastikan bahwa model machine learning dapat dilatih menggunakan sebagian besar data historis, dan dievaluasi menggunakan data yang belum pernah dilihat sebelumnya guna mengukur akurasi prediksinya secara efektif.",
            'lstm_flow': "Arsitektur Aliran LSTM",
            'lstm_cell': "Arsitektur Sel LSTM",
            'loss_curve': "Interpretasi Kurva Kerugian",
            'rmse': "Root Mean Squared Error (RMSE)",
            'mape': "Mean Absolute Percentage Error (MAPE)",
            'copyright': "© 2025 oleh Salsabila Fauziah",
            "data_used_description": "Data yang digunakan dalam proyek ini bersumber dari Yahoo Finance. Dataset mencakup harga saham historis dan volume perdagangan untuk perusahaan yang dipilih.",
            "time_used":"Agustus 2019 – Agustus 2024",
            "companies_label":"🏢Perusahaan yang Tersedia:",
            "companies_available_description": "Perusahaan yang tersedia untuk prediksi saham dalam proyek ini mencakup PT Telkom Indonesia (Persero) Tbk, Indosat Tbk PT, dan XL Axiata Tbk PT. Perusahaan-perusahaan ini dipilih karena signifikansi dan representasi mereka dalam sektor telekomunikasi Indonesia.",
            "data_test_label":"Data Uji",
            "data_train_label":"Data Latih",
            "lstm_detail_label":"Penjelasan Implementasi Long Short Term Memory",
            "how_lstm_works_label":"Bagaimana cara kerja LSTM:",
            "lstm_description":"Long Short-Term Memory (LSTM) adalah jenis arsitektur jaringan saraf berulang (RNN) yang mampu mempelajari ketergantungan jangka panjang. Ini sangat efektif untuk tugas prediksi deret waktu, seperti peramalan harga saham. Jaringan LSTM dirancang untuk mengingat informasi dalam jangka waktu lama, menjadikannya cocok untuk analisis data berurutan.",
            "inside_of_lstm_label":"Arsitektur Dalam Sel LSTM:",
            "inside_cell_lstm_description":"""Sel LSTM mengandung tiga gerbang: gerbang input, gerbang lupa, dan gerbang output. Gerbang ini mengontrol aliran informasi masuk dan keluar dari sel, memungkinkan sel untuk mempertahankan atau membuang informasi berdasarkan relevansinya dengan tugas prediksi saat ini.
            gerbang input menentukan seberapa banyak informasi baru yang harus disimpan, gerbang lupa memutuskan informasi apa yang harus dibuang, dan gerbang output mengontrol informasi apa yang dikirim ke lapisan berikutnya.
            Mekanisme penguncian ini memungkinkan jaringan LSTM untuk mempelajari pola kompleks dalam data berurutan dan membuat prediksi yang akurat.
            Dengan menggunakan LSTM, kita dapat secara efektif memodelkan ketergantungan temporal dalam pergerakan harga saham dan meningkatkan akurasi prediksi kita.
            """,
            "loss_curve_label": "Interpretasi Kurva Kerugian",
            "loss_curve_description": """Selama pelatihan model, kami melacak baik **kerugian pelatihan** maupun **kerugian validasi**. Indikator yang baik dari kinerja model adalah ketika kedua kurva kerugian secara bertahap menyatu dan tetap dekat. Ini menunjukkan bahwa model belajar dengan baik dan tidak mengalami underfitting atau overfitting.
             """,
             "loss_curve_highlight":"Semakin dekat garis kerugian pelatihan dan validasi, semakin stabil dan akurat modelnya.",
             "model_eval_label":"Evaluasi Kinerja Model",
             "model_eval_description":"Untuk mengevaluasi kinerja prediktif dari model Long Short-Term Memory (LSTM), kami menggunakan dua metrik umum: **Root Mean Squared Error (RMSE)** dan **Mean Absolute Percentage Error (MAPE)**. Metrik ini membantu menentukan seberapa baik prediksi model sesuai dengan harga saham aktual.",
             "rmse_description":"RMSE mengukur rata-rata besaran kesalahan antara nilai yang diprediksi dan nilai aktual. Ini menghukum kesalahan besar lebih parah daripada yang kecil.",
             "rmse_description_2":"RMSE yang lebih rendah menunjukkan kinerja model yang lebih baik, karena berarti nilai yang diprediksi lebih dekat dengan nilai aktual rata-rata.",
             "mape_description":"MAPE mengukur rata-rata persentase perbedaan antara nilai aktual dan yang diprediksi. Ini memberikan gambaran yang lebih jelas tentang akurasi prediksi dalam istilah persentase.",
             "mape_description_2":"MAPE yang lebih rendah menunjukkan bahwa model berkinerja baik dalam meminimalkan kesalahan prediksi relatif terhadap harga saham aktual.",
        }
    }
    
    return translations.get(language, {}).get(text_key, text_key)
