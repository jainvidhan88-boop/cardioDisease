import streamlit as st
import pandas as pd
import numpy as np
import pickle
from groq import Groq
import time
import streamlit.components.v1 as components
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="C.L.A.M. Analysis", page_icon="🧪", layout="wide")

# --- 2. THE "PRESTIGE" CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&family=Playfair+Display:ital,wght@0,700;1,700&display=swap');
    .main { background-color: #f1f5f9; }
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 50px; border-radius: 30px; text-align: center; color: white; margin-bottom: 30px;
    }
    .hero-title { font-family: 'Playfair Display', serif; font-size: 3.5em; margin: 0; }
    .stButton>button {
        background: #1e3a8a; color: white; border: none; padding: 18px;
        border-radius: 15px; font-weight: 600; width: 100%; transition: 0.3s;
    }
    .stButton>button:hover { background: #2563eb; transform: translateY(-2px); }
</style>
""", unsafe_allow_html=True)

# --- 3. THE BULLETPROOF LOADER ---
@st.cache_resource
def load_ml_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'heart_model.pkl')
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f: return pickle.load(f)
    return None

model = load_ml_model()

# --- 4. UI HEADER ---
st.markdown('<div class="hero-container"><div class="hero-title">CardioVascular Learning Analysis Model  C.L.A.M.</div><p style="opacity:0.8;">Machine Learning Interface for Cardiovascular Research</p></div>', unsafe_allow_html=True)

# --- 5. ENCODED INPUTS (3-Column Layout) ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Patient Data")
    p_name = st.text_input("Full Name", "Vidhan Jain")
    age = st.slider("Age", 18, 100, 45)
    sex = 1 if st.radio("Biological Sex", ["Male", "Female"]) == "Male" else 0

with col2:
    st.subheader("🩺 Vitals")
    bp = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    maxhr = st.slider("Max Heart Rate (BPM)", 60, 220, 150)
    exang = 1 if st.selectbox("Exercise Induced Angina", ["No", "Yes"]) == "Yes" else 0

with col3:
    st.subheader("📊 Diagnostics")
    cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal": 2, "Asymptomatic": 3}
    cp = cp_map[st.selectbox("Chest Pain Assessment", list(cp_map.keys()))]
    
    fbs = 1 if st.selectbox("Fasting Glucose > 120 mg/dL", ["Normal/False", "High/True"]) == "High/True" else 0
    
    thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
    thal = thal_map[st.selectbox("Thalassemia Scanning Result", list(thal_map.keys()))]

# Auto-set technical defaults for hidden parameters
restecg, oldpeak, slope, ca = 0, 1.0, 1, 0

# --- 6. EXECUTION ---
if st.button("GENERATE NEURAL SYNTHESIS REPORT"):
    if model:
        # Array of 13 features for the model
        features = np.array([[age, sex, cp, bp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal]])
        
        prob = model.predict_proba(features)[0][1]
        risk_score = "{:.1f}".format(prob * 100)
        status = "High Risk" if prob >= 0.5 else "Low Probability"
        accent = "#991b1b" if prob >= 0.5 else "#166534"
        bg_accent = "#fef2f2" if prob >= 0.5 else "#f0fdf4"

        

        with st.spinner("Processing through Neural Engine..."):
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            prompt = (
                f"Generate a professional cardiovascular analysis for {p_name}. "
                f"Result: {status} with {risk_score}% probability. "
                f"Use extremely formal clinical language. Include terms like 'ejection fraction' and 'coronary arteries'. "
                f"Write in one dense, professional paragraph. NO markdown formatting."
            )
            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
            clean_text = response.choices[0].message.content.replace("**", "").replace("*", "")

            # --- THE FINAL FORMATED REPORT ---
            report_html = f"""
            <div style="background: #FCFBF4; padding: 50px; border: 1px solid #d1d5db; max-width: 800px; margin: auto; font-family: 'Times New Roman', serif; color: #1a202c; box-shadow: 0 10px 25px rgba(0,0,0,0.1);">
                <div style="text-align: center; border-bottom: 2px solid #1e3a8a; padding-bottom: 20px; margin-bottom: 30px;">
                    <h1 style="margin:0; font-family: 'Arial', sans-serif; letter-spacing: 2px; color: #1e3a8a;">PREDICTIVE ANALYTICS REPORT</h1>
                    <p style="margin:5px; font-size: 0.75em; color: #64748b; letter-spacing: 1px;">CARDIOLOGY DATA SCIENCE PROJECT // SYSTEM V4.0</p>
                </div>

                <div style="background: {bg_accent}; border-left: 5px solid {accent}; padding: 15px; margin-bottom: 25px;">
                    <h4 style="margin:0; color: {accent}; font-family: Arial, sans-serif;">MODEL PREDICTION: {status.upper()}</h4>
                    <p style="margin:2px 0 0 0; font-weight: bold; font-family: Arial, sans-serif;">Confidence Score: {risk_score}%</p>
                </div>

                <div style="line-height: 1.7; text-align: justify; font-size: 1.1em;">{clean_text}</div>

                <div style="margin-top: 40px; padding: 15px; background: #f8fafc; border: 1px solid #e2e8f0; font-family: Arial, sans-serif; font-size: 0.75em; color: #475569;">
                    <strong>IMPORTANT NOTICE:</strong> This report is generated by an Artificial Intelligence model for educational and research purposes. 
                    It is <strong>NOT</strong> a medical diagnosis. The developer (Vidhan Jain) is a data science practitioner, not a licensed medical professional. 
                    Please consult a qualified cardiologist for medical advice.
                </div>

                <div style="margin-top: 30px; text-align: center; border-top: 1px solid #eee; padding-top: 15px;">
                    <p style="font-family: Arial, sans-serif; color: #94a3b8; font-size: 0.7em;">
                        Digital Audit Trail: CLAM-PRESTIGE-001 // Compiled by Vidhan Jain (Lead Developer)
                    </p>
                </div>
            </div>
            """
            components.html(report_html, height=800, scrolling=True)
    else:
        st.error("Missing Model File.")
