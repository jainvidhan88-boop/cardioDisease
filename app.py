import streamlit as st
import pandas as pd
import numpy as np
import pickle
from groq import Groq
import time
import streamlit.components.v1 as components
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="C.L.A.M. Prestige AI", page_icon="⚖️", layout="wide")

# --- 2. THE BEAUTY LAYER (CSS) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #f8fafc; }
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 50px; border-radius: 24px; text-align: center; color: white; margin-bottom: 30px;
    }
    .hero-title { font-size: 3em; font-weight: 700; letter-spacing: -1.5px; margin: 0; }
    .stButton>button {
        background: #1e3a8a; color: white; border-radius: 12px; padding: 12px; font-weight: 600; width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_ml_model():
    file_path = os.path.join(os.path.dirname(__file__), 'heart_model.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f: return pickle.load(f)
    return None

model = load_ml_model()

# --- 4. HEADER ---
st.markdown('<div class="hero-container"><div class="hero-title">C.L.A.M. PRESTIGE</div><p>Neural Cardiovascular Diagnostic Suite</p></div>', unsafe_allow_html=True)

# --- 5. ENCODED INPUTS ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 👤 Patient")
    p_name = st.text_input("Full Name", "Vidhan Jain")
    age = st.slider("Age", 18, 100, 45)
    # ENCODING: Male=1, Female=0
    sex_label = st.radio("Biological Sex", ["Male", "Female"])
    sex = 1 if sex_label == "Male" else 0

with col2:
    st.markdown("### 🩺 Vitals")
    bp = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    maxhr = st.slider("Max Heart Rate", 60, 220, 150)
    # ENCODING: Yes=1, No=0
    exang_label = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    exang = 1 if exang_label == "Yes" else 0

with col3:
    st.markdown("### 📊 Diagnostics")
    # ENCODING: Map common names to 0-3
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    cp_label = st.selectbox("Chest Pain Type", list(cp_map.keys()))
    cp = cp_map[cp_label]
    
    # ENCODING: Sugar > 120 (True=1, False=0)
    fbs_label = st.selectbox("Fasting Blood Sugar > 120mg/dL", ["False", "True"])
    fbs = 1 if fbs_label == "True" else 0
    
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal_label = st.selectbox("Thalassemia Result", list(thal_map.keys()))
    thal = thal_map[thal_label]

# Hidden technical values (defaults for the remaining 13 features)
restecg, oldpeak, slope, ca = 0, 1.0, 1, 0

st.markdown("<br>", unsafe_allow_html=True)
if st.button("PRODUCE CLINICAL SYNTHESIS"):
    if model:
        # Array of 13 features ordered correctly for the model
        input_data = np.array([[age, sex, cp, bp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal]])
        
        
        
        prob = model.predict_proba(input_data)[0][1]
        risk_score = "{:.1f}".format(prob * 100)
        status = "Positive" if prob >= 0.5 else "Negative"
        color = "#991b1b" if status == "Positive" else "#166534"

        with st.spinner("AI Analysis..."):
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            prompt = f"Professional summary for {p_name}. Heart Disease: {status} ({risk_score}%). Use clinical terms. No markdown."
            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
            
            # --- THE FINAL PRESTIGE REPORT ---
            report_html = f"""
            <div style="background: white; padding: 40px; border-radius: 20px; border: 1px solid #e2e8f0; box-shadow: 0 10px 30px rgba(0,0,0,0.05);">
                <h2 style="color: #0f172a; margin-top:0;">DIAGNOSTIC STATUS: <span style="color: {color};">{status.upper()}</span></h2>
                <hr style="border: 0; border-top: 1px solid #eee; margin: 20px 0;">
                <p style="font-size: 1.1em; line-height: 1.6;">{response.choices[0].message.content}</p>
                <div style="margin-top: 30px; padding: 15px; background: #f8fafc; border-radius: 10px; font-weight: 600;">
                    CONFIDENCE LEVEL: {risk_score}%
                </div>
            </div>
            """
            components.html(report_html, height=500)
    else:
        st.error("Model not loaded.")
