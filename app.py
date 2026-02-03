import streamlit as st
import pickle
import numpy as np

# Load model
try:
    model = pickle.load(open('heart_model.pkl', 'rb'))
except:
    st.error("Model file 'heart_model.pkl' not found.")

st.set_page_config(page_title="CardioGuard AI", page_icon="🫀", layout="wide")

# High-Contrast Dark Mode CSS
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; }
    h1, h2, h3, p, label, .stMarkdown { color: #FFFFFF !important; font-weight: 600; }
    
    /* Dynamic Metric Styling */
    [data-testid="stMetricValue"] { color: #00FFC8 !important; font-size: 1.8rem; }
    [data-testid="stMetricLabel"] { color: #B0B0B0 !important; }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00FFC8 0%, #008F7A 100%);
        color: #0E1117 !important;
        font-weight: bold; border: none; border-radius: 10px; height: 3.5em; width: 100%;
    }

    /* Professional Footer */
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #161B22; color: #00FFC8; text-align: center;
        padding: 8px; font-weight: bold; border-top: 1px solid #00FFC8; z-index: 1000;
    }
    </style>
    """, unsafe_allow_html=True)

tab_main, tab_about = st.tabs(["🚀 Diagnostic Tool", "📖 About the Model"])

with tab_main:
    st.title("🫀 CardioGuard AI: Live Diagnostic Suite")
    
    # Placeholder for dynamic metrics
    metric_cols = st.columns(4)

    st.markdown("---")

    # Data Inputs (Removed Form for Real-time Updates)
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 1, 100, 50)
        sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x==0 else "Male")
        bp = st.slider("Blood Pressure", 80, 200, 120)
        chol = st.slider("Cholesterol", 100, 600, 240)
    with col2:
        cp = st.selectbox("Chest Pain Severity", options=[1, 2, 3, 4], 
                          format_func=lambda x: {1:"Typical", 2:"Atypical", 3:"Non-anginal", 4:"Asymptomatic"}[x])
        fbs = st.radio("Fasting Sugar > 120", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        ekg = st.selectbox("EKG Status", options=[0, 1, 2], format_func=lambda x: ["Normal", "ST-T Wave", "LV Hypertrophy"][x])
        max_hr = st.number_input("Max Heart Rate", 60, 220, 150)
    with col3:
        angina = st.radio("Exercise Angina", options=[0, 1], format_func=lambda x: "No" if x==0 else "Yes", horizontal=True)
        oldpeak = st.number_input("ST Depression", 0.0, 6.2, 0.0, 0.1)
        slope = st.selectbox("ST Slope", options=[1, 2, 3], format_func=lambda x: ["Upsloping", "Flat", "Downsloping"][x-1])
        vessels = st.selectbox("Major Vessels", options=[0, 1, 2, 3])
        thal = st.selectbox("Thallium Result", options=[3, 6, 7], format_func=lambda x: {3:"Normal", 6:"Fixed", 7:"Reversible"}[x])

    # DYNAMIC UPDATES: Metrics Refresh Instantly
    metric_cols[0].metric("Age", f"{age}")
    metric_cols[1].metric("BP", f"{bp}", delta="High" if bp > 130 else "Normal", delta_color="inverse")
    metric_cols[2].metric("Cholesterol", f"{chol}", delta="Risk" if chol > 240 else "Good", delta_color="inverse")
    metric_cols[3].metric("ST Value", f"{oldpeak}")

    st.markdown("###")
    if st.button("🚀 GENERATE FINAL RISK ASSESSMENT"):
        features = np.array([[age, sex, cp, bp, chol, fbs, ekg, max_hr, angina, oldpeak, slope, vessels, thal]])
        prediction = model.predict(features)
        
        st.markdown("---")
        if prediction == 1:
            st.error("## ⚠️ HEART DISEASE DETECTED")
            st.warning("Immediate clinical review is recommended.")
        else:
            st.success("## ✅ NO HEART DISEASE DETECTED")
            st.info("Metrics suggest a low cardiac risk profile.")

with tab_about:
    st.header("Technical Specifications")
    st.markdown("""
    ### XGBoost Architecture
    This system utilizes **Extreme Gradient Boosting**, an ensemble method that builds decision trees sequentially to minimize prediction errors. It is currently one of the most efficient algorithms for structured medical data.
    
    ### Developer Credits
    **Created by: Vidhan Jain**  
    This project integrates advanced Machine Learning with a real-time responsive dashboard.
    """)

# Vidhan Jain Footer
st.markdown('<div class="footer">Developed by Vidhan Jain</div>', unsafe_allow_html=True)
