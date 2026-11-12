# Streamlit Klasifikasi Varietas Beras

1. Jalankan sel terakhir notebook penelitian untuk menghasilkan `best_model.joblib`.
2. Salin file tersebut ke folder yang sama dengan `app.py`.
3. Instal dependensi: `pip install -r requirements.txt`
4. Jalankan aplikasi: `streamlit run app.py`

Form otomatis mengikuti `feature_names` yang disimpan dalam model, sehingga urutan kolom tetap sesuai pelatihan. Model dan encoder tidak dilatih ulang saat prediksi.

Catatan: gunakan lingkungan Python dengan versi scikit-learn dan XGBoost yang kompatibel dengan lingkungan pelatihan. File joblib hanya boleh dimuat dari sumber yang dipercaya.
