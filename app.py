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

# --- 2. ADVANCED CSS (The Beauty Layer) ---
st.markdown("""
<style>
    /* Global Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #f8fafc;
    }

    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 60px;
        border-radius: 24px;
        text-align: center;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }

    .hero-title { font-size: 3.5em; font-weight: 700; margin-bottom: 10px; letter-spacing: -2px; }
    .hero-subtitle { font-size: 1.2em; opacity: 0.9; font-weight: 300; }

    /* Input Cards */
    div[data-testid="stExpander"] {
        border-radius: 15px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        background: white;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# --- 3. MODEL LOADING ---
@st.cache_resource
def load_ml_model():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, 'heart_model.pkl')
    if os.path.exists(file_path):
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except: return None
    return None

model = load_ml_model()

# --- 4. HEADER ---
st.markdown("""
    <div class="hero-container">
        <div class="hero-title">C.L.A.M. PRESTIGE</div>
        <div class="hero-subtitle">Advanced Cardiovascular Neural Inference & Diagnostic Engine</div>
    </div>
""", unsafe_allow_html=True)

# --- 5. LAYOUT: INPUTS ---
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("👤 Identity")
    p_name = st.text_input("Full Legal Name", "Vidhan Jain")
    age = st.slider("Age", 20, 90, 50)
    sex = st.selectbox("Biological Sex", ["Female", "Male"], index=1)

with col2:
    st.subheader("🩺 Vitals")
    bp = st.number_input("Resting BP (mmHg)", 80, 200, 120)
    chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 240)
    maxhr = st.slider("Max Heart Rate", 60, 220, 150)

with col3:
    st.subheader("📊 Clinical")
    cp = st.selectbox("Chest Pain", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [1, 2, 3])
    ca = st.selectbox("Vessels (0-3)", [0, 1, 2, 3])

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("EXECUTE NEURAL DIAGNOSIS")

# --- 6. ANALYSIS LOGIC ---
if predict_clicked:
    if model is None:
        st.error("System Error: Neural Weights ('heart_model.pkl') not initialized.")
    else:
        with st.spinner("Synchronizing Neural Clusters..."):
            # Prepare data
            sex_val = 1 if sex == "Male" else 0
            # Note: Add remaining 4 features (fbs, restecg, exang, oldpeak, slope) as defaults for prediction
            features = np.array([[age, sex_val, cp, bp, chol, 0, 0, maxhr, 0, 1.0, 1, ca, thal]])
            
            prob_raw = model.predict_proba(features)[0][1]
            risk_val = "{:.1f}".format(prob_raw * 100)
            status = "Positive" if prob_raw >= 0.5 else "Negative"
            
            # Colors for report
            accent = "#991b1b" if status == "Positive" else "#166534"
            bg_accent = "#fef2f2" if status == "Positive" else "#f0fdf4"

            # Groq AI Synthesis
            client = Groq(api_key=st.secrets["GROQ_API_KEY"])
            prompt = (f"Act as a prestige AI Physician. Provide a high-end diagnostic summary for {p_name}. "
                      f"Result: {status} ({risk_val}%). Use professional, clear language. No markdown.")
            
            response = client.chat.completions.create(messages=[{"role": "user", "content": prompt}], model="llama-3.3-70b-versatile")
            ai_content = response.choices[0].message.content.replace('\n', '<br>')

            # --- THE PRESTIGE REPORT HTML ---
            report_html = f"""
            <div style="background: white; padding: 60px; border-radius: 30px; border: 1px solid #e2e8f0; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.1); font-family: 'Inter', sans-serif; color: #1e293b;">
                <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 2px solid #f1f5f9; padding-bottom: 30px; margin-bottom: 40px;">
                    <div>
                        <h1 style="margin:0; font-weight: 700; color: #0f172a; letter-spacing: -1px;">CLINICAL CERTIFICATE</h1>
                        <p style="margin:0; color: #64748b; font-size: 0.9em;">A.I. DIAGNOSTIC DIVISION // REF: {int(time.time())}</p>
                    </div>
                    <div style="text-align: right;">
                        <span style="background: {bg_accent}; color: {accent}; padding: 8px 16px; border-radius: 99px; font-weight: 600; font-size: 0.8em; border: 1px solid {accent}44;">
                            STATUS: {status.upper()}
                        </span>
                    </div>
                </div>

                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 40px; margin-bottom: 40px;">
                    <div style="background: #f8fafc; padding: 25px; border-radius: 20px;">
                        <p style="margin:0; color: #64748b; font-size: 0.8em; font-weight: 600;">CONFIDENCE SCORE</p>
                        <h2 style="margin:0; font-size: 2.5em; color: #0f172a;">{risk_val}<span style="font-size: 0.5em; color: #94a3b8;">%</span></h2>
                    </div>
                    <div style="background: #f8fafc; padding: 25px; border-radius: 20px;">
                        <p style="margin:0; color: #64748b; font-size: 0.8em; font-weight: 600;">PATIENT RECORD</p>
                        <h2 style="margin:0; font-size: 1.5em; color: #0f172a; padding-top: 10px;">{p_name}</h2>
                    </div>
                </div>

                <div style="line-height: 1.8; color: #334155; font-size: 1.1em; text-align: justify;">
                    {ai_content}
                </div>

                <div style="margin-top: 60px; border-top: 1px solid #f1f5f9; padding-top: 30px; font-size: 0.8em; color: #94a3b8; text-align: center;">
                    THIS IS A COMPUTER-GENERATED DOCUMENT. NO SIGNATURE REQUIRED.<br>
                    <strong>C.L.A.M. ENGINE V4.0 // PRESTIGE EDITION</strong>
                </div>
            </div>
            """
            components.html(f"<div style='background-color:#f8fafc; padding: 20px;'>{report_html}</div>", height=850, scrolling=True)

            st.download_button("📥 DOWNLOAD OFFICIAL TRANSCRIPT", ai_content, "Official_CLAM_Report.txt")
