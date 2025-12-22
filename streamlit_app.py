import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Telco Churn Prediction App",
    page_icon="",
    layout="wide"
)

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_model():
    # Memuat file pkl yang berisi model dan preprocessing
    return joblib.load('churn_model_assets.pkl')

try:
    assets = load_model()
    model = assets['model']
    scaler = assets['scaler']
    encoders = assets['label_encoders']
    feature_names = assets['feature_names']
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'churn_model_assets.pkl' ada di folder yang sama. Error: {e}")
    st.stop()

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("📝 Input Data Pelanggan")

def get_user_input():
    # Fitur Numerik
    tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)
    senior_citizen = st.sidebar.selectbox("Senior Citizen (0=No, 1=Yes)", [0, 1])

    # Fitur Kategorikal
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", 
                                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    data = {
        'gender': gender, 'SeniorCitizen': senior_citizen, 'Partner': partner,
        'Dependents': dependents, 'tenure': tenure, 'PhoneService': phone_service,
        'MultipleLines': multiple_lines, 'InternetService': internet_service,
        'OnlineSecurity': online_security, 'OnlineBackup': online_backup,
        'DeviceProtection': device_protection, 'TechSupport': tech_support,
        'StreamingTV': streaming_tv, 'StreamingMovies': streaming_movies,
        'Contract': contract, 'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method, 'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

input_df = get_user_input()

# --- MAIN PAGE ---
st.title("📊 Telco Customer Churn Predictor")
st.markdown("Aplikasi ini menggunakan model **Machine Learning** terbaik untuk memprediksi potensi pelanggan berhenti berlangganan.")

# Row 1: Data Info
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Profil Pelanggan")
    st.table(input_df.iloc[:, :7]) # Tampilkan sebagian data saja agar tidak penuh

    # --- PREPROCESSING INPUT ---
    df_ready = input_df.copy()
    
    # 1. Label Encoding
    for col, encoder in encoders.items():
        if col in df_ready.columns:
            df_ready[col] = encoder.transform(df_ready[col])
    
    # 2. Scaling
    cols_to_scale = assets['cols_to_scale']
    df_ready[cols_to_scale] = scaler.transform(df_ready[cols_to_scale])
    
    # 3. Reorder Columns
    df_ready = df_ready[feature_names]

    # --- PREDICTION ---
    prediction = model.predict(df_ready)
    prediction_proba = model.predict_proba(df_ready)

    st.subheader("Hasil Prediksi")
    if prediction[0] == 1:
        st.error("🚨 HASIL: Pelanggan ini kemungkinan besar akan **CHURN** (Berhenti).")
    else:
        st.success("✅ HASIL: Pelanggan ini kemungkinan besar akan **RETAIN** (Tetap Langganan).")

with col2:
    st.subheader("Confidence Level")
    prob_churn = prediction_proba[0][1] * 100
    prob_retain = prediction_proba[0][0] * 100
    
    # Visualisasi Gauge sederhana atau Bar
    st.metric("Probabilitas Churn", f"{prob_churn:.2f}%")
    st.progress(int(prob_churn))
    
    st.metric("Probabilitas Bertahan", f"{prob_retain:.2f}%")
    st.progress(int(prob_retain))

# --- DATA VISUALIZATION (PENDUKUNG) ---
st.divider()
st.subheader("💡 Analisis Fitur Input")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Grafik Monthly Charges
    fig, ax = plt.subplots()
    sns.barplot(x=['Input Pelanggan', 'Rata-rata Global'], 
                y=[input_df['MonthlyCharges'][0], 64.7], palette='coolwarm', ax=ax)
    ax.set_ylabel("Monthly Charges ($)")
    ax.set_title("Biaya Bulanan vs Rata-rata")
    st.pyplot(fig)

with viz_col2:
    # Info Kontrak
    st.write(f"**Tipe Kontrak:** {input_df['Contract'][0]}")
    if input_df['Contract'][0] == "Month-to-month":
        st.warning("Kontrak Month-to-month memiliki risiko churn paling tinggi berdasarkan data historis.")
    else:
        st.info("Kontrak jangka panjang membantu mengurangi risiko churn.")

st.markdown("---")
st.caption("Developed by Muhammad Za'im Muzakki | NIM: A11.2022.14023")
