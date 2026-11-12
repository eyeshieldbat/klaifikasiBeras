from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Klasifikasi Varietas Beras', page_icon='🌾')
st.title('🌾 Klasifikasi Varietas Beras')
st.write('Prediksi varietas **Cammeo** atau **Osmancik** berdasarkan fitur morfologi.')

MODEL_PATH = Path(__file__).parent / 'best_model.joblib'

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.warning('File best_model.joblib belum ditemukan. Salin hasil ekspor notebook ke folder aplikasi ini.')
    st.stop()

try:
    artifact = load_model()
    model = artifact['model']
    encoder = artifact['label_encoder']
    features = artifact['feature_names']
except Exception as exc:
    st.error(f'Gagal membaca model: {exc}')
    st.stop()

st.subheader('Masukkan fitur morfologi')
st.caption('Isi nilai hasil pengukuran citra beras, bukan perkiraan visual. Satuan harus sama dengan dataset pelatihan.')
with st.form('prediction_form'):
    values = {}
    cols = st.columns(2)
    for i, feature in enumerate(features):
        with cols[i % 2]:
            values[feature] = st.number_input(feature.replace('_', ' '), min_value=0.0, value=None, format='%.6f', key=feature, placeholder='Masukkan nilai')
    submitted = st.form_submit_button('Prediksi Varietas', type='primary', use_container_width=True)

if submitted:
    if any(v is None for v in values.values()):
        st.error('Harap isi seluruh fitur sebelum melakukan prediksi.')
    else:
        sample = pd.DataFrame([values], columns=features)
        try:
            pred = model.predict(sample)
            label = encoder.inverse_transform(pred)[0]
            st.success(f'Hasil prediksi: **{label}**')
            st.dataframe(sample, hide_index=True, use_container_width=True)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(sample)[0]
                st.subheader('Probabilitas model')
                prob_df = pd.DataFrame({'Varietas': encoder.classes_, 'Probabilitas': probabilities})
                st.bar_chart(prob_df.set_index('Varietas'))
                st.caption('Probabilitas model bukan jaminan ketepatan prediksi.')
        except Exception as exc:
            st.error(f'Prediksi gagal: {exc}')
