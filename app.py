import streamlit as st
import pandas as pd
import numpy as np
import pickle
from groq import Groq
import time
import streamlit.components.v1 as components
import os

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(page_title="C.L.A.M. AI Physician", page_icon="👨‍⚕️", layout="centered")

# --- 2. UI STYLING ---
st.markdown("""
<style>
    .main-title { color: #1e3a8a; text-align: center; font-weight: bold; margin-bottom: 5px; }
    .made-by { text-align: center; font-size: 0.9em; color: #64748b; margin-bottom: 25px; }
    .accuracy-card {
        background: #f0f7ff; border: 1px solid #cce3ff;
        color: #1e3a8a; padding: 20px; border-radius: 12px; text-align: center; margin-bottom: 25px;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING (Fixed Pathing & Global Variable) ---
@st.cache_resource
def load_ml_model():
    # Use absolute path to ensure the server finds it regardless of directory
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'heart_model.pkl')
    
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except:
        return None

# CRITICAL: This line fixes the NameError
model = load_ml_model()

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("📋 Patient Profile")
    p_name = st.text_input("Patient Name", "Vidhan Jain")
    st.divider()
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Sex", ["Female", "Male"], index=1)
    cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3], format_func=lambda x: ["Typical Angina", "Atypical Angina", "Non-anginal", "Asymptomatic"][x])
    bp = st.number_input("Resting Blood Pressure", 80, 200, 120)
    chol = st.number_input("Cholesterol", 100, 600, 240)
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG Results", [0, 1, 2])
    maxhr = st.slider("Max Heart Rate", 60, 220, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    slope = st.selectbox("ST Slope", [0, 1, 2])
    ca = st.selectbox("Vessels (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3], format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x-1])
    
    predict_clicked = st.button("RUN AI DIAGNOSIS", type="primary", use_container_width=True)

# --- 5. MAIN LOGIC ---
if not predict_clicked:
    st.markdown("<h1 class='main-title'>C.L.A.M. Heart Health AI</h1>", unsafe_allow_html=True)
    st.markdown("<p class='made-by'>Created by Vidhan Jain</p>", unsafe_allow_html=True)
    st.markdown("""<div class="accuracy-card"><h3>Diagnostic Accuracy: 88.0%</h3></div>""", unsafe_allow_html=True)
    
    # Pre-check for the user
    if model is None:
        st.error("⚠️ SYSTEM ALERT: 'heart_model.pkl' not detected. Prediction functionality is disabled.")
else:
    try:
        sex_val = 1 if sex == "Male" else 0
        features = np.array([[age, sex_val, cp, bp, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca, thal]])
        
        # Diagnosis Logic
        if model is not None:
            prob_raw = model.predict_proba(features)[0][1]
            risk_val = "{:.1f}".format(prob_raw * 100)
            status = "Elevated Risk Detected" if prob_raw >= 0.5 else "No Significant Risk Detected"
            status_color = "#fef2f2" if prob_raw >= 0.5 else "#f0fdf4"
            text_color = "#991b1b" if prob_raw >= 0.5 else "#166534"
            border_color = "#fecaca" if prob_raw >= 0.5 else "#bbf7d0"
        else:
            risk_val, status, status_color, text_color, border_color = "0.0", "SYSTEM ERROR: MODEL MISSING", "#eee", "#333", "#ccc"

        with st.spinner("AI Analysis in Progress..."):
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            
            prompt = (
                f"You are the C.L.A.M. AI Medical Assistant. Write a formal clinical summary for {p_name}. "
                f"Stats: Age {age}, BP {bp}, Cholesterol {chol}. "
                f"ML Prediction: {status} ({risk_val}% Confidence). "
                f"STRICT INSTRUCTION: Do NOT use any markdown formatting like asterisks (** or *) or hashtags. "
                f"Write in plain, professional sentences. Use clear section headers like 'CLINICAL FINDINGS' and 'RECOMMENDATIONS'."
            )
            
            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
            
            # Remove Markdown symbols for clean paper look
            clean_content = response.choices[0].message.content.replace("**", "").replace("*", "").replace("#", "")
            ai_html = clean_content.replace('\n', '<br>')

            # --- THE FINAL FORMATED REPORT ---
            report_html = f"""
            <html>
            <body style="margin:0; padding:0; background-color: #FCFBF4;">
                <div style="background-color: #FCFBF4; color: #1e293b; padding: 50px; border: 1px solid #e2e8f0; font-family: 'Times New Roman', Times, serif; line-height: 1.7;">
                    
                    <table width="100%" style="border-bottom: 2px solid #1e3a8a; margin-bottom: 20px;">
                        <tr>
                            <td style="padding-bottom: 10px;">
                                <h1 style="margin:0; color: #1e3a8a; font-family: Arial, sans-serif;">CLINICAL ANALYSIS REPORT</h1>
                                <p style="margin:0; font-size: 0.85em; color: #64748b; font-family: Arial, sans-serif;">Generated by Cardiovascular Learning Analysis Model (C.L.A.M.)</p>
                            </td>
                            <td style="text-align: right; vertical-align: bottom; font-family: Arial, sans-serif; font-size: 0.8em; color: #64748b;">
                                REPORT ID: {int(time.time())}
                            </td>
                        </tr>
                    </table>

                    <div style="background-color: {status_color}; color: {text_color}; border: 1px solid {border_color}; padding: 15px; text-align: center; margin-bottom: 30px;">
                        <h2 style="margin:0; font-family: Arial, sans-serif; font-size: 1.2em;">DIAGNOSTIC STATUS: {status.upper()}</h2>
                        <p style="margin: 5px 0 0 0; font-family: Arial, sans-serif;">Statistical Confidence Score: {risk_val}%</p>
                    </div>

                    <table width="100%" style="margin-bottom: 30px; font-family: Arial, sans-serif; font-size: 0.9em; background: #fff; padding: 15px; border: 1px solid #eee;">
                        <tr>
                            <td><strong>PATIENT NAME:</strong> {p_name}</td>
                            <td style="text-align: right;"><strong>DATE:</strong> {time.strftime("%d %B %Y")}</td>
                        </tr>
                        <tr>
                            <td><strong>AGE:</strong> {age}</td>
                            <td style="text-align: right;"><strong>REFERRAL:</strong> Automated AI System</td>
                        </tr>
                    </table>

                    <div style="font-size: 1.15em; text-align: justify;">
                        {ai_html}
                    </div>
                    
                    <div style="margin-top: 50px; padding: 20px; background-color: #f8fafc; border: 1px solid #cbd5e1; font-size: 0.8em; color: #475569; font-family: Arial, sans-serif;">
                        <strong>OFFICIAL DISCLAIMER:</strong> This document is an automated synthesis of data patterns based on an 88% accurate model. It does not constitute a legal medical diagnosis. 
                        Final clinical decisions should be made in consultation with a licensed medical professional.
                    </div>
                    
                    <p style="text-align: center; font-size: 0.8em; color: #94a3b8; margin-top: 40px; font-family: Arial, sans-serif; border-top: 1px solid #eee; padding-top: 10px;">
                        Digital Signature: C.L.A.M. Physician Core // Lead Developer: Vidhan Jain
                    </p>
                </div>
            </body>
            </html>
            """
            
            components.html(report_html, height=900, scrolling=True)

            st.divider()
            st.download_button("📥 Save Clinical Report (.txt)", f"CLINICAL REPORT\nPatient: {p_name}\n\n{clean_content}", f"Report_{p_name}.txt", use_container_width=True)
            if st.button("Initialize New Analysis"): st.rerun()

    except Exception as e:
        st.error(f"Error during synthesis: {e}")
