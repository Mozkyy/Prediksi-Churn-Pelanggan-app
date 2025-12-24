import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Telco Churn Prediction App",
    page_icon="📊",
    layout="wide"
)

# --- HELPER FUNCTIONS untuk Error Handling ---
def safe_label_encode(value, encoder, column_name):
    """
    Safely encode a value using Label Encoder.
    If value not in classes, return default (first class).
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        st.warning(f"⚠️ Nilai '{value}' untuk kolom '{column_name}' tidak dikenali. Menggunakan default: '{encoder.classes_[0]}'")
        return 0

def get_valid_options(encoders, column_name):
    """Get valid options from encoder for a column"""
    if column_name in encoders:
        return list(encoders[column_name].classes_)
    return []

# --- LOAD MODEL & ASSETS ---
@st.cache_resource
def load_model():
    """Memuat model dan assets dari file pkl"""
    try:
        assets = joblib.load('streamlit_model_balanced.pkl')
        return assets
    except FileNotFoundError:
        st.error("❌ File 'streamlit_model_balanced.pkl' tidak ditemukan!")
        st.info("Pastikan file berada di folder yang sama dengan streamlit_app.py")
        st.stop()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        st.stop()

# Load assets
assets = load_model()
model = assets['model']
scaler = assets.get('scaler')
encoders = assets['label_encoders']
feature_names = assets['feature_names']
model_info = assets.get('model_info', {})
class_weight_info = assets.get('class_weight_info', {})

# --- SIDEBAR: INPUT USER ---
st.sidebar.header("📝 Input Data Pelanggan")
st.sidebar.markdown("Masukkan informasi pelanggan untuk prediksi churn")

def get_user_input():
    """Fungsi untuk mengambil input dari user dengan validasi"""
    
    st.sidebar.subheader("👤 Demografi")
    
    # Gender dengan valid options
    gender_options = get_valid_options(encoders, 'gender')
    gender = st.sidebar.selectbox("Gender", gender_options if gender_options else ["Male", "Female"])
    
    senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya (65+)" if x == 1 else "Tidak")
    
    # Partner & Dependents dengan valid options
    partner_options = get_valid_options(encoders, 'Partner')
    partner = st.sidebar.selectbox("Partner", partner_options if partner_options else ["Yes", "No"])
    
    dependents_options = get_valid_options(encoders, 'Dependents')
    dependents = st.sidebar.selectbox("Dependents", dependents_options if dependents_options else ["Yes", "No"])
    
    st.sidebar.subheader("📞 Layanan")
    
    # Phone Service
    phone_service_options = get_valid_options(encoders, 'PhoneService')
    phone_service = st.sidebar.selectbox("Phone Service", phone_service_options if phone_service_options else ["Yes", "No"])
    
    # Multiple Lines
    multiple_lines_options = get_valid_options(encoders, 'MultipleLines')
    multiple_lines = st.sidebar.selectbox("Multiple Lines", multiple_lines_options if multiple_lines_options else ["No phone service", "No", "Yes"])
    
    # Internet Service
    internet_service_options = get_valid_options(encoders, 'InternetService')
    internet_service = st.sidebar.selectbox("Internet Service", internet_service_options if internet_service_options else ["DSL", "Fiber optic", "No"])
    
    # Online Security
    online_security_options = get_valid_options(encoders, 'OnlineSecurity')
    online_security = st.sidebar.selectbox("Online Security", online_security_options if online_security_options else ["No", "Yes", "No internet service"])
    
    # Online Backup
    online_backup_options = get_valid_options(encoders, 'OnlineBackup')
    online_backup = st.sidebar.selectbox("Online Backup", online_backup_options if online_backup_options else ["No", "Yes", "No internet service"])
    
    # Device Protection
    device_protection_options = get_valid_options(encoders, 'DeviceProtection')
    device_protection = st.sidebar.selectbox("Device Protection", device_protection_options if device_protection_options else ["No", "Yes", "No internet service"])
    
    # Tech Support
    tech_support_options = get_valid_options(encoders, 'TechSupport')
    tech_support = st.sidebar.selectbox("Tech Support", tech_support_options if tech_support_options else ["No", "Yes", "No internet service"])
    
    # Streaming TV
    streaming_tv_options = get_valid_options(encoders, 'StreamingTV')
    streaming_tv = st.sidebar.selectbox("Streaming TV", streaming_tv_options if streaming_tv_options else ["No", "Yes", "No internet service"])
    
    # Streaming Movies
    streaming_movies_options = get_valid_options(encoders, 'StreamingMovies')
    streaming_movies = st.sidebar.selectbox("Streaming Movies", streaming_movies_options if streaming_movies_options else ["No", "Yes", "No internet service"])
    
    st.sidebar.subheader("💳 Akun & Billing")
    
    # Fitur Numerik
    tenure = st.sidebar.slider("Tenure (Bulan)", 0, 72, 12, help="Lama berlangganan dalam bulan")
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0, step=0.5, help="Biaya bulanan")
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 800.0, step=1.0, help="Total biaya keseluruhan")
    
    # Contract
    contract_options = get_valid_options(encoders, 'Contract')
    contract = st.sidebar.selectbox("Contract", contract_options if contract_options else ["Month-to-month", "One year", "Two year"])
    
    # Paperless Billing
    paperless_billing_options = get_valid_options(encoders, 'PaperlessBilling')
    paperless_billing = st.sidebar.selectbox("Paperless Billing", paperless_billing_options if paperless_billing_options else ["Yes", "No"])
    
    # Payment Method
    payment_method_options = get_valid_options(encoders, 'PaymentMethod')
    payment_method = st.sidebar.selectbox(
        "Payment Method", 
        payment_method_options if payment_method_options else ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )

    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame([data])

# Get user input
input_df = get_user_input()

# --- MAIN PAGE ---
st.title("📊 Telco Customer Churn Predictor")
st.markdown("Aplikasi ini menggunakan model **Machine Learning** dengan class_weight='balanced' untuk memprediksi potensi pelanggan berhenti berlangganan.")

# Display model info
if model_info:
    with st.expander("ℹ️ Informasi Model"):
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Model", model_info.get('model_name', 'N/A'))
        with col_info2:
            metrics = model_info.get('metrics', {})
            st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        with col_info3:
            st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        with col_info4:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        
        if class_weight_info:
            st.info(f"🔧 Imbalance Handling: class_weight='{class_weight_info.get('method', 'balanced')}'")

st.markdown("---")

# Row 1: Data Info
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("👤 Profil Pelanggan")
    
    # Tampilkan data dalam format yang lebih baik
    profile_data = {
        "Atribut": ["Gender", "Senior Citizen", "Partner", "Dependents", "Tenure (bulan)", "Monthly Charges", "Total Charges"],
        "Nilai": [
            input_df['gender'].iloc[0],
            "Ya" if input_df['SeniorCitizen'].iloc[0] == 1 else "Tidak",
            input_df['Partner'].iloc[0],
            input_df['Dependents'].iloc[0],
            input_df['tenure'].iloc[0],
            f"${input_df['MonthlyCharges'].iloc[0]:.2f}",
            f"${input_df['TotalCharges'].iloc[0]:.2f}"
        ]
    }
    st.table(pd.DataFrame(profile_data))

with col2:
    st.subheader("📋 Info Layanan")
    services_data = {
        "Layanan": ["Phone Service", "Internet Service", "Contract", "Payment Method"],
        "Detail": [
            input_df['PhoneService'].iloc[0],
            input_df['InternetService'].iloc[0],
            input_df['Contract'].iloc[0],
            input_df['PaymentMethod'].iloc[0]
        ]
    }
    st.table(pd.DataFrame(services_data))

# --- PREPROCESSING INPUT dengan Error Handling ---
st.markdown("---")
st.subheader("🔄 Processing Prediction...")

try:
    with st.spinner("Memproses data..."):
        df_ready = input_df.copy()
        
        # 1. Label Encoding dengan Safe Transform
        for col, encoder in encoders.items():
            if col in df_ready.columns:
                # Safe encoding
                df_ready[col] = df_ready[col].apply(
                    lambda x: safe_label_encode(x, encoder, col)
                )
        
        # 2. Scaling (jika ada scaler)
        if scaler is not None:
            cols_to_scale = assets.get('cols_to_scale', ['tenure', 'MonthlyCharges', 'TotalCharges'])
            if cols_to_scale:
                df_ready[cols_to_scale] = scaler.transform(df_ready[cols_to_scale])
        
        # 3. Reorder Columns sesuai feature_names
        df_ready = df_ready[feature_names]

        # --- PREDICTION ---
        prediction = model.predict(df_ready)[0]
        prediction_proba = model.predict_proba(df_ready)[0]
    
    # --- HASIL PREDIKSI ---
    st.markdown("---")
    
    result_col1, result_col2 = st.columns([2, 1])
    
    with result_col1:
        st.subheader("🎯 Hasil Prediksi")
        
        # Determine prediction result (handle both numeric and string)
        is_churn = (prediction == 1) or (prediction == 'Yes')
        
        if is_churn:
            st.error("### 🚨 HASIL: Pelanggan ini kemungkinan besar akan **CHURN** (Berhenti)")
            st.markdown("""
            **Rekomendasi Tindakan:**
            - 🎁 Tawarkan promo atau diskon khusus
            - 📞 Hubungi pelanggan untuk feedback
            - 🎯 Berikan loyalty reward
            - 📧 Kirim retention campaign
            """)
        else:
            st.success("### ✅ HASIL: Pelanggan ini kemungkinan besar akan **RETAIN** (Tetap Langganan)")
            st.markdown("""
            **Rekomendasi Tindakan:**
            - ⭐ Pertahankan kualitas layanan
            - 🚀 Tawarkan upgrade/cross-sell
            - 💌 Kirim appreciation message
            - 📊 Monitor perubahan behavior
            """)
    
    with result_col2:
        st.subheader("📊 Confidence Level")
        
        # Handle probability (might be in different order)
        prob_churn = prediction_proba[1] * 100 if len(prediction_proba) > 1 else prediction_proba[0] * 100
        prob_retain = prediction_proba[0] * 100 if len(prediction_proba) > 1 else (100 - prob_churn)
        
        # Visualisasi Metrics
        st.metric("Probabilitas Churn", f"{prob_churn:.2f}%", delta=None)
        st.progress(int(min(prob_churn, 100)))
        
        st.metric("Probabilitas Bertahan", f"{prob_retain:.2f}%", delta=None)
        st.progress(int(min(prob_retain, 100)))
        
        # Risk Level
        if prob_churn > 70:
            st.error("🔴 Risk Level: TINGGI")
        elif prob_churn > 50:
            st.warning("🟡 Risk Level: SEDANG")
        else:
            st.success("🟢 Risk Level: RENDAH")

    # --- DATA VISUALIZATION (PENDUKUNG) ---
    st.divider()
    st.subheader("💡 Analisis Fitur Input")

    viz_col1, viz_col2, viz_col3 = st.columns(3)

    with viz_col1:
        # Grafik Monthly Charges
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        categories = ['Input\nPelanggan', 'Rata-rata\nGlobal']
        values = [input_df['MonthlyCharges'].iloc[0], 64.76]
        colors = ['#e74c3c' if is_churn else '#2ecc71', '#3498db']
        
        bars = ax1.bar(categories, values, color=colors, edgecolor='black', linewidth=1.5)
        ax1.set_ylabel("Monthly Charges ($)", fontsize=11, fontweight='bold')
        ax1.set_title("Biaya Bulanan vs Rata-rata", fontsize=12, fontweight='bold')
        ax1.set_ylim(0, max(values) * 1.2)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'${height:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig1)

    with viz_col2:
        # Info Kontrak
        st.markdown("**📋 Analisis Kontrak**")
        contract_type = input_df['Contract'].iloc[0]
        st.info(f"**Tipe Kontrak:** {contract_type}")
        
        if contract_type == "Month-to-month":
            st.warning("⚠️ Kontrak Month-to-month memiliki risiko churn paling tinggi berdasarkan data historis.")
        elif contract_type == "One year":
            st.info("ℹ️ Kontrak 1 tahun memiliki risiko churn sedang.")
        else:
            st.success("✅ Kontrak 2 tahun memiliki risiko churn paling rendah!")

    with viz_col3:
        # Tenure Analysis
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        tenure_val = input_df['tenure'].iloc[0]
        
        # Create tenure categories
        if tenure_val < 12:
            tenure_cat = "< 1 tahun"
            risk = "Tinggi"
            color = '#e74c3c'
        elif tenure_val < 36:
            tenure_cat = "1-3 tahun"
            risk = "Sedang"
            color = '#f39c12'
        else:
            tenure_cat = "> 3 tahun"
            risk = "Rendah"
            color = '#2ecc71'
        
        ax2.barh(['Tenure'], [tenure_val], color=color, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel("Bulan", fontsize=11, fontweight='bold')
        ax2.set_title(f"Tenure: {tenure_val} bulan ({tenure_cat})", fontsize=12, fontweight='bold')
        ax2.text(tenure_val/2, 0, f'{tenure_val} bulan', 
                ha='center', va='center', fontsize=10, fontweight='bold', color='white')
        
        plt.tight_layout()
        st.pyplot(fig2)
        
        st.info(f"**Risiko berdasarkan Tenure:** {risk}")

    # Additional insights
    st.divider()
    st.subheader("📈 Insight Tambahan")
    
    insight_col1, insight_col2 = st.columns(2)
    
    with insight_col1:
        st.markdown("**🔍 Faktor Risiko Terdeteksi:**")
        risk_factors = []
        
        if input_df['Contract'].iloc[0] == "Month-to-month":
            risk_factors.append("- Kontrak month-to-month")
        if input_df['PaymentMethod'].iloc[0] == "Electronic check":
            risk_factors.append("- Payment method: Electronic check")
        if input_df['tenure'].iloc[0] < 12:
            risk_factors.append("- Tenure kurang dari 1 tahun")
        if input_df['MonthlyCharges'].iloc[0] > 80:
            risk_factors.append("- Monthly charges tinggi (>$80)")
        if input_df['InternetService'].iloc[0] == "Fiber optic":
            risk_factors.append("- Internet service: Fiber optic (biaya tinggi)")
        
        if risk_factors:
            for factor in risk_factors:
                st.warning(factor)
        else:
            st.success("✅ Tidak ada faktor risiko mayor terdeteksi")
    
    with insight_col2:
        st.markdown("**💪 Faktor Protektif Terdeteksi:**")
        protective_factors = []
        
        if input_df['Contract'].iloc[0] in ["One year", "Two year"]:
            protective_factors.append("- Kontrak jangka panjang")
        if input_df['tenure'].iloc[0] > 24:
            protective_factors.append("- Tenure lebih dari 2 tahun")
        if input_df['Partner'].iloc[0] == "Yes":
            protective_factors.append("- Memiliki partner")
        if input_df['Dependents'].iloc[0] == "Yes":
            protective_factors.append("- Memiliki dependents")
        if input_df['TechSupport'].iloc[0] == "Yes":
            protective_factors.append("- Berlangganan Tech Support")
        
        if protective_factors:
            for factor in protective_factors:
                st.success(factor)
        else:
            st.info("ℹ️ Pertimbangkan untuk meningkatkan engagement pelanggan")

except Exception as e:
    st.error("❌ Terjadi error saat melakukan prediksi!")
    st.error(f"**Error:** {str(e)}")
    
    with st.expander("🐛 Debug Information"):
        st.write("**Input DataFrame:**")
        st.dataframe(input_df)
        
        st.write("**Expected Features:**")
        st.write(feature_names)
        
        st.write("**Available Encoders:**")
        st.write(list(encoders.keys()))
        
        import traceback
        st.code(traceback.format_exc())

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>Telco Customer Churn Predictor</strong></p>
    <p>Developed by Muhammad Za'im Muzakki | NIM: A11.2022.14023</p>
    <p>Model: class_weight='balanced' | Framework: Streamlit + Scikit-learn</p>
</div>
""", unsafe_allow_html=True)
