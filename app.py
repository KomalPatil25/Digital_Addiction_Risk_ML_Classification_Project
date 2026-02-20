import streamlit as st
import numpy as np
import joblib
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Digital Addiction Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ================= CENTER WIDTH =================
st.markdown("""
<style>
.block-container {
    max-width: 900px;
    margin: auto;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rf.pkl")

model = joblib.load(model_path)

# ================= BACKGROUND STYLE =================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #2193b0, #6dd5ed);
}

.header-box {
    text-align:center;
    color:white;
    margin-bottom:25px;
}

.header-box h1 {
    margin-bottom:5px;
}

.header-box p {
    opacity:0.9;
}

.pred-card {
    padding:25px;
    border-radius:14px;
    text-align:center;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)

# ================= HEADER =================
st.markdown("""
<div class="header-box">
    <h1>🧠 AI Digital Addiction Risk Analyzer</h1>
    <p>Understanding digital habits and their impact on everyday wellbeing</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ================= INPUT GRID =================
st.subheader("👤 User Lifestyle Profile")

c1, c2, c3 = st.columns(3)

with c1:
    Screen_Time_Hours = st.number_input("Screen Time (hrs)", 0.0, 16.0, 6.0)
    Social_Media_Hours = st.number_input("Social Media (hrs)", 0.0, 12.0, 3.0)

with c2:
    Sleep_Hours = st.number_input("Sleep Hours", 0.0, 12.0, 7.0)
    Physical_Activity_Min = st.number_input("Activity (min)", 0, 300, 60)

with c3:
    App_Unlocks_Per_Day = st.number_input("App Unlocks (Per Day)", 0, 300, 60)

st.divider()

# ================= PREDICTION =================
center = st.columns([1,2,1])[1]

with center:
    if st.button("✨ Analyze Addiction Risk", use_container_width=True):

        features = np.array([[ 
            Screen_Time_Hours,
            Social_Media_Hours,
            Sleep_Hours,
            Physical_Activity_Min,
            App_Unlocks_Per_Day
        ]])

        prediction = model.predict(features)[0]

        if prediction == "Low":
            st.markdown("""
            <div class="pred-card" style="background:#E8F5E9;">
                <h2 style="color:#2E7D32;">🟢 Low Addiction Risk</h2>
                <p>Balanced digital usage and healthy lifestyle indicators.</p>
            </div>
            """, unsafe_allow_html=True)

        elif prediction == "Medium":
            st.markdown("""
            <div class="pred-card" style="background:#FFF8E1;">
                <h2 style="color:#F9A825;">🟡 Moderate Addiction Risk</h2>
                <p>Elevated digital engagement detected — monitor behavior.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="pred-card" style="background:#FFEBEE;">
                <h2 style="color:#C62828;">🔴 High Addiction Risk</h2>
                <p>High dependency pattern detected — lifestyle intervention recommended.</p>
            </div>
            """, unsafe_allow_html=True)

        st.caption("Prediction: " + prediction)

st.divider()

# ================= FOOTER =================
st.markdown("""
<div style="text-align:center;color:white;font-size:13px;">
AI Model: Random Forest • Digital Behavior Dataset • Streamlit Interface
    • Komal Patil        
</div>
""", unsafe_allow_html=True)

