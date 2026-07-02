import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(
    page_title="Smart Hospital Patient Navigator",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { display: none; }
.stDeployButton { display: none; }
footer { visibility: hidden; }
.block-container { padding-top: 0 !important; padding-bottom: 2rem !important; max-width: 1100px !important; }
div[data-testid="stForm"] { border: none; padding: 0; }

div.stButton > button, div.stFormSubmitButton > button {
    background: linear-gradient(135deg, #1a56db, #1e429f) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 16px !important; font-weight: 600 !important;
    width: 100% !important; letter-spacing: 0.02em !important;
    box-shadow: 0 4px 14px rgba(26, 86, 219, 0.35) !important;
}

div.stButton > button:hover, div.stFormSubmitButton > button:hover {
    background: linear-gradient(135deg, #1e429f, #1a56db) !important;
}

div[data-testid="stCheckbox"] label {
    font-size: 14px !important; font-weight: 500 !important; color: #374151 !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    with open('hospital_model.pkl', 'rb') as f:
        return pickle.load(f)


bundle = load_model()
model = bundle['model']
scaler = bundle['scaler']
features = bundle['features']
cols_to_scale = bundle['cols_to_scale']
dept_map_inv = bundle['dept_map_inv']
gender_map = bundle['gender_map']
temp_map = bundle['temp_map']
hr_map = bundle['hr_map']
dur_map = bundle['dur_map']
cc_map = bundle['cc_map']

DEPT_INFO = {
    'Respiratory Medicine': {
        'icon': '🫁',
        'color': '#0284c7',
        'bg': '#e0f2fe',
        'border': '#7dd3fc',
        'desc': 'Specialises in conditions affecting the lungs and airways.',
        'next': ['Visit Level 2, Wing B', 'Estimated wait: 15-25 min', 'Please wear a mask']
    },
    'Cardiology': {
        'icon': '❤️',
        'color': '#dc2626',
        'bg': '#fee2e2',
        'border': '#fca5a5',
        'desc': 'Specialises in heart and cardiovascular conditions.',
        'next': ['Visit Level 3, Wing A', 'Estimated wait: 20-30 min', 'Bring any previous ECG reports']
    },
    'Gastroenterology': {
        'icon': '🧪',
        'color': '#d97706',
        'bg': '#fef3c7',
        'border': '#fcd34d',
        'desc': 'Specialises in digestive system and abdominal conditions.',
        'next': ['Visit Level 1, Wing C', 'Estimated wait: 10-20 min', 'Avoid eating before consultation']
    },
    'Neurology': {
        'icon': '🧠',
        'color': '#7c3aed',
        'bg': '#ede9fe',
        'border': '#c4b5fd',
        'desc': 'Specialises in brain, spine, and nervous system conditions.',
        'next': ['Visit Level 4, Wing A', 'Estimated wait: 25-35 min', 'Bring list of current medications']
    },
    'General Medicine': {
        'icon': '🩺',
        'color': '#059669',
        'bg': '#d1fae5',
        'border': '#6ee7b7',
        'desc': 'Handles general health concerns and non-specialist conditions.',
        'next': ['Visit Level 1, Wing A', 'Estimated wait: 10-15 min', 'Registration Desk is open 24/7']
    },
    'Dermatology': {
        'icon': '🔬',
        'color': '#b45309',
        'bg': '#fef9c3',
        'border': '#fde68a',
        'desc': 'Specialises in conditions affecting the skin, hair, and nails.',
        'next': ['Visit Level 2, Wing D', 'Estimated wait: 15-25 min', 'Avoid applying creams beforehand']
    },
}

# Hero Header
st.markdown("""
<div style="background:linear-gradient(135deg, #1e3a8a 0%, #1a56db 60%, #0ea5e9 100%);
            padding: 3rem 2rem 2.5rem; margin: -1rem -1rem 2rem; text-align: center;
            border-radius: 0 0 20px 20px;">
    <div style="font-size: 32px; font-weight: 700; color: white;">
        🏥 Smart Hospital Patient Navigator
    </div>
    <div style="font-size: 15px; color: #dbeafe; margin-top: 6px;">
        Answer a few questions and we'll point you to the right department.
    </div>
</div>
""", unsafe_allow_html=True)

# Form
with st.form("triage_form"):

    # Section 1 - Symptoms
    st.markdown("""
    <div style="background:#f0f9ff;border:1px solid #bae6fd; border-radius:14px;
            padding:20px 24px; margin-bottom:20px;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#0284c7;color:white;border-radius:8px;
                    padding:4px 10px; font-size:12px;font-weight:600;">1</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">What are your symptoms?</span>
            <span style="font-size:13px;color:#6b7280;font-style:italic;">select all that apply</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        fever = st.checkbox("Fever")
        cough = st.checkbox("Cough")
        headache = st.checkbox("Headache")
    with c2:
        chest_pain = st.checkbox("Chest Pain")
        stomach_pain = st.checkbox("Stomach Pain")
        shortness_breath = st.checkbox("Shortness of Breath")
    with c3:
        nausea_vomiting = st.checkbox("Nausea/Vomiting")
        dizziness = st.checkbox("Dizziness")
        skin_rash = st.checkbox("Skin Rash")

    # Section 2 - Chief complaint & duration
    st.markdown("""
    <div style="background:#fdf4ff;border:1px solid #e9d5ff; border-radius:14px;
            padding:20px 24px; margin:20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#7c3aed;color:white;border-radius:8px;
                    padding:4px 10px; font-size:12px;font-weight:600;">2</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">
                What's the main concern, and how long have you had it?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_cc, col_dur = st.columns(2)
    with col_cc:
        chief_complaint = st.selectbox("Chief Complaint", options=list(cc_map.keys()))
    with col_dur:
        duration = st.selectbox("Duration", options=list(dur_map.keys()))

    # Section 3 - Severity
    st.markdown("""
    <div style="background:#fff7ed;border:1px solid #fed7aa; border-radius:14px;
            padding:20px 24px; margin:20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#ea580c;color:white;border-radius:8px;
                    padding:4px 10px; font-size:12px;font-weight:600;">3</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">How severe do your vitals feel?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_temp, col_hr = st.columns(2)
    with col_temp:
        temperature_level = st.selectbox("Temperature", options=list(temp_map.keys()), index=1)
    with col_hr:
        heart_rate_level = st.selectbox("Heart Rate", options=list(hr_map.keys()), index=1)

    # Section 4 - Medical history
    st.markdown("""
    <div style="background:#f0fdf4;border:1px solid #bbf7d0; border-radius:14px;
            padding:20px 24px; margin:20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#059669;color:white;border-radius:8px;
                    padding:4px 10px; font-size:12px;font-weight:600;">4</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">Any relevant medical history?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    h1, h2, h3 = st.columns(3)
    with h1:
        asthma = st.checkbox("Asthma")
    with h2:
        hypertension = st.checkbox("Hypertension")
    with h3:
        heart_disease = st.checkbox("Heart Disease")

    # Section 5 - Patient Info
    st.markdown("""
    <div style="background:#f8fafc;border:1px solid #e2e8f0; border-radius:14px;
            padding:20px 24px; margin:20px 0;">
        <div style="display:flex;align-items:center;gap:10px;">
            <span style="background:#475569;color:white;border-radius:8px;
                    padding:4px 10px; font-size:12px;font-weight:600;">5</span>
            <span style="font-size:16px;font-weight:600;color:#0c4a6e;">Patient Information</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_age, col_gen = st.columns(2)
    with col_age:
        age = st.number_input("Age", min_value=1, max_value=100, value=35)
    with col_gen:
        gender = st.selectbox("Gender", options=list(gender_map.keys()))

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Get AI Recommendation →")

# When the form is submitted
if submitted:
    patient = pd.DataFrame([{
        'age': age,
        'gender': gender_map.get(gender, 0),
        'fever': int(fever),
        'cough': int(cough),
        'headache': int(headache),
        'chest_pain': int(chest_pain),
        'stomach_pain': int(stomach_pain),
        'shortness_breath': int(shortness_breath),
        'nausea_vomitting': int(nausea_vomiting),
        'dizziness': int(dizziness),
        'skin_rash': int(skin_rash),
        'temperature_level': temp_map.get(temperature_level, 1),
        'heart_rate_level': hr_map.get(heart_rate_level, 1),
        'duration': dur_map.get(duration, 1),
        'asthma': int(asthma),
        'hypertension': int(hypertension),
        'heart_disease': int(heart_disease),
        'chief_complaint': cc_map.get(chief_complaint, 9)
    }])

    patient_scaled = patient.copy()
    patient_scaled[cols_to_scale] = scaler.transform(patient[cols_to_scale])

    pred = model.predict(patient_scaled[features])[0]
    proba = model.predict_proba(patient_scaled[features])[0]
    dept_name = dept_map_inv[pred]
    confidence = proba[pred] * 100
    info = DEPT_INFO.get(dept_name, {
        'icon': '🏥', 'color': '#1a56db', 'bg': '#eff6ff', 'border': '#bfdbfe',
        'desc': 'Please proceed to the registration desk for guidance.',
        'next': ['Visit the Registration Desk', 'Estimated wait: 10-15 min']
    })

    next_steps = "".join(
        f'<li style="margin-bottom:6px;">{step}</li>' for step in info['next']
    )

    st.markdown(f"""
<div style="background:{info['bg']};border:1.6px solid {info['border']};
            border-radius:16px;padding:28px 32px;margin-top:8px;">
    <div style="font-size:44px">{info['icon']}</div>
    <div style="font-size:24px;font-weight:700;color:{info['color']};margin-top:4px;">{dept_name}</div>
    <div style="font-size:14px;color:#4b5563;margin-top:6px;">{info['desc']}</div>
    <div style="font-size:14px;font-weight:600;color:#111827;margin-top:16px;">
        Confidence: {confidence:.1f}%
    </div>
    <div style="font-size:14px;font-weight:600;color:#111827;margin-top:16px;">Next steps</div>
    <ul style="font-size:14px;color:#374151;margin-top:8px;padding-left:20px;">
        {next_steps}
    </ul>
</div>
""", unsafe_allow_html=True)
