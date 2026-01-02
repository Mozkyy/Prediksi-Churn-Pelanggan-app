import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Telco Churn Prediction App",
    page_icon="📱",
    layout="wide"
)

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_model():
    # Memuat file pkl yang berisi model terbaik
    return joblib.load('streamlit_model_balanced.pkl')

try:
    assets = load_model()
    model = assets['model']
    scaler = assets['scaler']
    encoders = assets['label_encoders']
    feature_names = assets['feature_names']
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan file 'streamlit_model_balanced.pkl' ada di folder yang sama. Error: {e}")
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
    
    # Pastikan TotalCharges dalam format yang benar (float, bukan string)
    # Konversi ke float jika belum
    if df_ready['TotalCharges'].dtype == 'object':
        df_ready['TotalCharges'] = pd.to_numeric(df_ready['TotalCharges'], errors='coerce').fillna(0)
    
    # Pastikan semua kolom numerik dalam format float
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in df_ready.columns:
            df_ready[col] = df_ready[col].astype(float)
    
    # 1. Label Encoding untuk kolom kategorikal
    for col, encoder in encoders.items():
        if col in df_ready.columns:
            try:
                # Pastikan data dalam format string untuk encoding
                df_ready[col] = df_ready[col].astype(str)
                df_ready[col] = encoder.transform(df_ready[col])
            except Exception as e:
                st.warning(f"Warning saat encoding kolom {col}: {e}")
                # Gunakan nilai default jika terjadi error
                df_ready[col] = 0
    
    # 2. Scaling untuk kolom numerik
    cols_to_scale = assets['cols_to_scale']
    try:
        df_ready[cols_to_scale] = scaler.transform(df_ready[cols_to_scale])
    except Exception as e:
        st.error(f"Error saat scaling: {e}")
        st.stop()
    
    # 3. Reorder Columns sesuai dengan feature_names dari model
    df_ready = df_ready[feature_names]

    # --- PREDICTION ---
    try:
        prediction = model.predict(df_ready)
        prediction_proba = model.predict_proba(df_ready)

        st.subheader("Hasil Prediksi")
        if prediction[0] == 1:
            st.error("🚨 HASIL: Pelanggan ini kemungkinan besar akan **CHURN** (Berhenti).")
        else:
            st.success("✅ HASIL: Pelanggan ini kemungkinan besar akan **RETAIN** (Tetap Langganan).")
    except Exception as e:
        st.error(f"Error saat melakukan prediksi: {e}")
        st.stop()

with col2:
    st.subheader("Confidence Level")
    prob_churn = prediction_proba[0][1] * 100
    prob_retain = prediction_proba[0][0] * 100
    
    # Visualisasi Gauge sederhana atau Bar
    st.metric("Probabilitas Churn", f"{prob_churn:.2f}%")
    st.progress(int(prob_churn) / 100)
    
    st.metric("Probabilitas Bertahan", f"{prob_retain:.2f}%")
    st.progress(int(prob_retain) / 100)

# --- REKOMENDASI TINDAKAN ---
st.divider()
st.subheader("💡 Rekomendasi Tindakan")

if prediction[0] == 1:
    st.markdown("### 🎯 Strategi Retensi yang Disarankan:")
    
    recommendations = []
    
    # Analisis berdasarkan kontrak
    if input_df['Contract'][0] == "Month-to-month":
        recommendations.append("🎁 Tawarkan promo atau diskon khusus untuk upgrade ke kontrak jangka panjang (1 atau 2 tahun)")
    
    # Analisis berdasarkan biaya bulanan
    if input_df['MonthlyCharges'][0] > 70:
        recommendations.append("💰 Pertimbangkan untuk menawarkan paket bundling dengan harga lebih kompetitif")
    
    # Analisis berdasarkan layanan internet
    if input_df['InternetService'][0] == "Fiber optic":
        recommendations.append("🌐 Pastikan kualitas layanan Fiber optic optimal untuk mempertahankan pelanggan premium")
    
    # Analisis berdasarkan layanan tambahan
    if input_df['TechSupport'][0] == "No":
        recommendations.append("🛠️ Tawarkan trial gratis untuk Tech Support sebagai nilai tambah")
    
    if input_df['OnlineSecurity'][0] == "No":
        recommendations.append("🔒 Berikan penawaran khusus untuk layanan Online Security")
    
    for rec in recommendations:
        st.write(rec)
else:
    st.success("✅ Pelanggan ini memiliki risiko churn rendah. Pertahankan kualitas layanan yang baik!")

# --- DATA VISUALIZATION (PENDUKUNG) ---
st.divider()
st.subheader("📈 Analisis Fitur Input")

viz_col1, viz_col2 = st.columns(2)

with viz_col1:
    # Grafik Monthly Charges
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Input Pelanggan', 'Rata-rata Global'], 
                y=[input_df['MonthlyCharges'][0], 64.7], palette='coolwarm', ax=ax)
    ax.set_ylabel("Monthly Charges ($)", fontsize=12)
    ax.set_title("Biaya Bulanan vs Rata-rata", fontsize=14, fontweight='bold')
    st.pyplot(fig)

with viz_col2:
    # Grafik Tenure
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Input Pelanggan', 'Rata-rata Global'], 
                y=[input_df['tenure'][0], 32], palette='viridis', ax=ax2)
    ax2.set_ylabel("Tenure (Bulan)", fontsize=12)
    ax2.set_title("Lama Berlangganan vs Rata-rata", fontsize=14, fontweight='bold')
    st.pyplot(fig2)

# --- INFO KONTRAK ---
st.divider()
st.subheader("📋 Informasi Detail")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.write(f"**Tipe Kontrak:** {input_df['Contract'][0]}")
    if input_df['Contract'][0] == "Month-to-month":
        st.warning("⚠️ Kontrak Month-to-month memiliki risiko churn paling tinggi berdasarkan data historis.")
    else:
        st.info("✅ Kontrak jangka panjang membantu mengurangi risiko churn.")

with info_col2:
    st.write(f"**Internet Service:** {input_df['InternetService'][0]}")
    if input_df['InternetService'][0] == "Fiber optic":
        st.info("🚀 Pelanggan Fiber optic cenderung menggunakan lebih banyak layanan.")
    elif input_df['InternetService'][0] == "No":
        st.warning("📵 Tidak menggunakan layanan internet.")

with info_col3:
    st.write(f"**Payment Method:** {input_df['PaymentMethod'][0]}")
    if input_df['PaymentMethod'][0] == "Electronic check":
        st.warning("⚠️ Electronic check memiliki korelasi lebih tinggi dengan churn.")
    else:
        st.success("✅ Metode pembayaran otomatis meningkatkan retensi.")

st.markdown("---")
st.caption("Developed by Muhammad Za'im Muzakki | NIM: A11.2022.14023")
