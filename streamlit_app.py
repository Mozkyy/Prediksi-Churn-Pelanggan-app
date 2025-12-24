import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Prediksi Churn Pelanggan",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL
# ============================================================================

@st.cache_resource
def load_model():
    """Load model dan assets dari file"""
    try:
        assets = joblib.load('streamlit_model_balanced.pkl')
        return assets
    except FileNotFoundError:
        st.error("❌ File model tidak ditemukan!")
        st.info("Pastikan file 'streamlit_model_balanced.pkl' ada di folder yang sama")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()

# Load assets
assets = load_model()
model = assets['model']
scaler = assets.get('scaler')
label_encoders = assets['label_encoders']
feature_names = assets['feature_names']
model_info = assets.get('model_info', {})

# ============================================================================
# HELPER FUNCTIONS - FIX ERROR HANDLING
# ============================================================================

def safe_label_encode(value, encoder, column_name):
    """
    Safely encode a value using Label Encoder.
    If value not in classes, return default (first class).
    """
    try:
        return encoder.transform([value])[0]
    except ValueError:
        # Value not recognized, use default
        st.warning(f"⚠️ '{value}' tidak dikenali untuk '{column_name}'. Menggunakan default: '{encoder.classes_[0]}'")
        return 0

def get_valid_options(column_name):
    """Get valid options from encoder for a column"""
    if column_name in label_encoders:
        return list(label_encoders[column_name].classes_)
    return []

def validate_and_encode_input(input_dict):
    """
    Validate and encode input dictionary.
    Returns encoded dataframe.
    """
    # Create dataframe from input
    df = pd.DataFrame([input_dict])
    
    # Encode categorical columns
    for col, encoder in label_encoders.items():
        if col in df.columns:
            # Safe encoding
            df[col] = df[col].apply(
                lambda x: safe_label_encode(x, encoder, col)
            )
    
    return df

# ============================================================================
# SIDEBAR - MODEL INFO
# ============================================================================

with st.sidebar:
    st.title("ℹ️ Info Model")
    
    if model_info:
        st.metric("Model", model_info.get('model_name', 'N/A'))
        st.metric("Scenario", model_info.get('scenario', 'N/A'))
        
        metrics = model_info.get('metrics', {})
        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.4f}")
        st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
        st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
        
        # Class weight info
        if 'imbalance_handling' in model_info:
            st.info(f"⚖️ Imbalance Handling: {model_info['imbalance_handling']['method']}")
    
    st.write("---")
    st.caption("Developed with ❤️ using Streamlit")

# ============================================================================
# MAIN APP
# ============================================================================

st.title("🔮 Prediksi Churn Pelanggan Telekomunikasi")
st.write("Prediksi apakah pelanggan akan berhenti berlangganan (churn) atau tidak.")
st.write("---")

# ============================================================================
# INPUT FORM
# ============================================================================

st.write("### 📝 Masukkan Data Pelanggan")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    
    # ========================================================================
    # COLUMN 1: Demographics
    # ========================================================================
    with col1:
        st.write("**👤 Demografi**")
        
        gender = st.selectbox(
            "Gender",
            options=get_valid_options('gender'),
            help="Jenis kelamin pelanggan"
        )
        
        senior_citizen = st.selectbox(
            "Senior Citizen",
            options=[0, 1],
            format_func=lambda x: "Ya" if x == 1 else "Tidak",
            help="Apakah pelanggan berusia 65 tahun ke atas?"
        )
        
        partner = st.selectbox(
            "Partner",
            options=get_valid_options('Partner'),
            help="Apakah pelanggan memiliki pasangan?"
        )
        
        dependents = st.selectbox(
            "Dependents",
            options=get_valid_options('Dependents'),
            help="Apakah pelanggan memiliki tanggungan?"
        )
        
        tenure = st.number_input(
            "Tenure (bulan)",
            min_value=0,
            max_value=100,
            value=12,
            help="Lama berlangganan dalam bulan"
        )
    
    # ========================================================================
    # COLUMN 2: Services
    # ========================================================================
    with col2:
        st.write("**📞 Layanan Telepon & Internet**")
        
        phone_service = st.selectbox(
            "Phone Service",
            options=get_valid_options('PhoneService'),
            help="Berlangganan layanan telepon?"
        )
        
        multiple_lines = st.selectbox(
            "Multiple Lines",
            options=get_valid_options('MultipleLines'),
            help="Memiliki multiple phone lines?"
        )
        
        internet_service = st.selectbox(
            "Internet Service",
            options=get_valid_options('InternetService'),
            help="Tipe layanan internet"
        )
        
        online_security = st.selectbox(
            "Online Security",
            options=get_valid_options('OnlineSecurity'),
            help="Berlangganan Online Security?"
        )
        
        online_backup = st.selectbox(
            "Online Backup",
            options=get_valid_options('OnlineBackup'),
            help="Berlangganan Online Backup?"
        )
        
        device_protection = st.selectbox(
            "Device Protection",
            options=get_valid_options('DeviceProtection'),
            help="Berlangganan Device Protection?"
        )
        
        tech_support = st.selectbox(
            "Tech Support",
            options=get_valid_options('TechSupport'),
            help="Berlangganan Tech Support?"
        )
    
    # ========================================================================
    # COLUMN 3: Streaming & Account
    # ========================================================================
    with col3:
        st.write("**📺 Streaming & Akun**")
        
        streaming_tv = st.selectbox(
            "Streaming TV",
            options=get_valid_options('StreamingTV'),
            help="Berlangganan Streaming TV?"
        )
        
        streaming_movies = st.selectbox(
            "Streaming Movies",
            options=get_valid_options('StreamingMovies'),
            help="Berlangganan Streaming Movies?"
        )
        
        contract = st.selectbox(
            "Contract",
            options=get_valid_options('Contract'),
            help="Tipe kontrak pelanggan"
        )
        
        paperless_billing = st.selectbox(
            "Paperless Billing",
            options=get_valid_options('PaperlessBilling'),
            help="Menggunakan paperless billing?"
        )
        
        payment_method = st.selectbox(
            "Payment Method",
            options=get_valid_options('PaymentMethod'),
            help="Metode pembayaran"
        )
        
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=0.0,
            max_value=200.0,
            value=50.0,
            step=0.5,
            help="Biaya bulanan dalam USD"
        )
        
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=10000.0,
            value=float(tenure * monthly_charges),
            step=1.0,
            help="Total biaya keseluruhan dalam USD"
        )
    
    # Submit button
    submitted = st.form_submit_button(
        "🔮 Prediksi Churn",
        use_container_width=True,
        type="primary"
    )

# ============================================================================
# PREDICTION
# ============================================================================

if submitted:
    # Create input dictionary
    input_data = {
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
    
    try:
        with st.spinner("🔄 Memproses prediksi..."):
            # Validate and encode
            df_encoded = validate_and_encode_input(input_data)
            
            # Scale numeric features (if scaler exists)
            if scaler is not None:
                cols_to_scale = assets.get('cols_to_scale', ['tenure', 'MonthlyCharges', 'TotalCharges'])
                if cols_to_scale:
                    df_encoded[cols_to_scale] = scaler.transform(df_encoded[cols_to_scale])
            
            # Ensure correct column order
            df_ready = df_encoded[feature_names]
            
            # Predict
            prediction = model.predict(df_ready)[0]
            proba = model.predict_proba(df_ready)[0]
            
            # Get probability for each class
            # Assuming classes are ['No', 'Yes']
            prob_no_churn = proba[0]
            prob_churn = proba[1]
        
        # ====================================================================
        # DISPLAY RESULTS
        # ====================================================================
        
        st.write("---")
        st.write("### 🎯 Hasil Prediksi")
        
        # Main prediction result
        if prediction == 'Yes':
            st.error("### ⚠️ PELANGGAN BERPOTENSI CHURN!")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Status Prediksi",
                    "CHURN",
                    delta="High Risk",
                    delta_color="inverse"
                )
            with col2:
                st.metric(
                    "Probabilitas Churn",
                    f"{prob_churn:.1%}",
                    delta=None
                )
            with col3:
                st.metric(
                    "Confidence Level",
                    "High" if prob_churn > 0.7 else "Medium",
                    delta=None
                )
            
            # Progress bar
            st.write("**Churn Risk Score:**")
            st.progress(float(prob_churn), text=f"Risk Level: {prob_churn:.1%}")
            
            # Recommendations
            st.write("### 💡 Rekomendasi Tindakan")
            st.warning("""
            **Segera lakukan tindakan retention:**
            
            1. 🎁 **Tawarkan Promo Khusus**
               - Diskon untuk perpanjangan kontrak
               - Upgrade gratis ke paket premium
            
            2. 📞 **Hubungi Pelanggan**
               - Survey kepuasan pelanggan
               - Identifikasi masalah yang dihadapi
            
            3. 🎯 **Program Loyalty**
               - Reward points untuk pelanggan setia
               - Cashback atau voucher
            
            4. 📧 **Retention Campaign**
               - Email personalized dengan benefits
               - SMS reminder tentang value proposition
            """)
            
        else:
            st.success("### ✅ PELANGGAN KEMUNGKINAN BERTAHAN")
            
            # Metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Status Prediksi",
                    "TIDAK CHURN",
                    delta="Low Risk",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Probabilitas Bertahan",
                    f"{prob_no_churn:.1%}",
                    delta=None
                )
            with col3:
                st.metric(
                    "Confidence Level",
                    "High" if prob_no_churn > 0.7 else "Medium",
                    delta=None
                )
            
            # Progress bar
            st.write("**Retention Score:**")
            st.progress(float(prob_no_churn), text=f"Retention Level: {prob_no_churn:.1%}")
            
            # Recommendations
            st.write("### 💡 Rekomendasi Tindakan")
            st.info("""
            **Pertahankan dan tingkatkan engagement:**
            
            1. ⭐ **Maintain Service Quality**
               - Monitor kepuasan secara berkala
               - Pastikan service tetap excellent
            
            2. 🚀 **Upsell Opportunities**
               - Tawarkan add-on services
               - Upgrade ke paket yang lebih tinggi
            
            3. 💌 **Appreciation Program**
               - Thank you message berkala
               - Birthday/anniversary rewards
            
            4. 📊 **Monitor Behavior**
               - Track usage patterns
               - Early warning jika ada perubahan
            """)
        
        # Detailed probability breakdown
        st.write("### 📊 Detail Probabilitas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Probabilitas Tidak Churn:**")
            st.progress(float(prob_no_churn))
            st.write(f"{prob_no_churn:.2%}")
        
        with col2:
            st.write("**Probabilitas Churn:**")
            st.progress(float(prob_churn))
            st.write(f"{prob_churn:.2%}")
        
        # Risk level indicator
        if prob_churn > 0.7:
            risk_level = "🔴 TINGGI"
            risk_color = "red"
        elif prob_churn > 0.5:
            risk_level = "🟡 SEDANG"
            risk_color = "orange"
        else:
            risk_level = "🟢 RENDAH"
            risk_color = "green"
        
        st.write(f"**Risk Level: {risk_level}**")
        
        # Debug info (collapsible)
        with st.expander("🔍 Debug Info (untuk developer)"):
            st.write("**Input Data (Original):**")
            st.json(input_data)
            
            st.write("**Input Data (Encoded):**")
            st.dataframe(df_encoded)
            
            st.write("**Input Data (Ready for Prediction):**")
            st.dataframe(df_ready)
            
            st.write("**Prediction Details:**")
            st.write(f"- Prediction: {prediction}")
            st.write(f"- Probabilities: {proba}")
            st.write(f"- Model: {type(model).__name__}")
    
    except Exception as e:
        st.error("❌ Terjadi error saat melakukan prediksi!")
        st.error(f"**Error Message:** {str(e)}")
        
        # Detailed error info
        with st.expander(" Detail Error (untuk debugging)"):
            import traceback
            st.code(traceback.format_exc())
            
            st.write("**Input Data:**")
            st.json(input_data)
            
            st.write("**Expected Features:**")
            st.write(feature_names)
            
            st.write("**Available Encoders:**")
            st.write(list(label_encoders.keys()))
            
            st.write("**Encoder Classes:**")
            for col, encoder in label_encoders.items():
                st.write(f"- {col}: {list(encoder.classes_)}")

# ============================================================================
# FOOTER
# ============================================================================

st.write("---")
st.caption(" Tip: Pastikan semua input sesuai dengan opsi yang tersedia untuk hasil prediksi yang akurat.")
st.caption(" Developed by Muhammad Za'im Muzakki - A11.2022.14023")
