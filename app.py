# import streamlit as st
# import numpy as np
# import pickle

# with open("rf_model.pkl", "rb") as file:
#     model = pickle.load(file)


# st.set_page_config(
#     page_title="Digital Addiction Risk Predictor",
#     page_icon="📱",
#     layout="centered"
# )


# st.title("📱 Digital Addiction Risk Predictor")
# st.write("Enter user details to assess digital addiction risk:")

# st.subheader("Enter User Behavior Data")

# Age = st.number_input("Age", min_value=0)

# Screen_Time_Hours = st.number_input("Screen_Time_Hours", min_value=0.0)

# Social_Media_Hours = st.number_input("Social_Media_Hours", min_value=0.0)

# Sleep_Hours = st.number_input("Sleep_Hours", min_value=0.0)

# Study_Work_Hours  = st.number_input("Study_Work_Hours ", min_value=0.0)

# Physical_Activity_Min = st.number_input("Physical_Activity_Min", min_value=0.0)

# App_Unlocks_Per_Day = st.number_input("App_Unlocks_Per_Day", min_value=0.0)

# Mood_Score = st.number_input("Mood_Score", min_value=0)


# if st.button("Predict Addiction Risk"):
#     input_data = np.array([[Age, Screen_Time_Hours, Social_Media_Hours,	Sleep_Hours, Study_Work_Hours, Physical_Activity_Min,	App_Unlocks_Per_Day, Mood_Score	]])
#     prediction = model.predict(input_data)[0]
    
#     # st.subheader("Prediction Result")

#     if prediction == "Low":
#         st.success("🟢 Low Addiction Risk")
#     elif prediction == "Medium":
#         st.warning("🟡 Medium Addiction Risk")
#     else:
#         st.error("🔴 High Addiction Risk")

#     st.write("Prediction:", prediction)






import streamlit as st
import numpy as np
import pickle
import os

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI Digital Addiction Analyzer",
    page_icon="🧠",
    layout="wide"
)

# ================= LOAD MODEL =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rf_model.pkl")

with open(model_path, "rb") as file:
    model = pickle.load(file)

# ================= BACKGROUND STYLE =================
st.markdown("""
<style>
.stApp {
    # background-image: url("https://images.unsplash.com/photo-1519389950473-47ba0277781c");
    background-color: 990099;
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.header-box {
    padding: 30px;
    border-radius: 15px;
    background: linear-gradient(135deg,#667eea,#764ba2);
    color: white;
    text-align: center;
    margin-bottom: 25px;
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

# ================= INPUT GRID =================
st.subheader("👤 User Lifestyle Profile")

c1, c2, c3, c4 = st.columns(4)

with c1:
    Age = st.number_input("Age", min_value=10, max_value=70, value=25)
    Sleep_Hours = st.number_input("Sleep Hours", min_value=0.0, max_value=12.0, value=7.0)

with c2:
    Screen_Time_Hours = st.number_input("Screen Time", min_value=0.0, max_value=16.0, value=6.0)
    Social_Media_Hours = st.number_input("Social Media", min_value=0.0, max_value=12.0, value=3.0)

with c3:
    Study_Work_Hours = st.number_input("Study/Work", min_value=0.0, max_value=16.0, value=6.0)
    Physical_Activity_Min = st.number_input("Activity (min)", min_value=0, max_value=300, value=60)

with c4:
    App_Unlocks_Per_Day = st.number_input("App Unlocks", min_value=0, max_value=300, value=80)
    Mood_Score = st.number_input("Mood Score", min_value=1, max_value=10, value=5)

st.divider()

# ================= PREDICTION CENTER =================
center = st.columns([1,2,1])[1]

with center:
    if st.button("✨ Analyze Addiction Risk", use_container_width=True):

        features = np.array([[ 
            Age,
            Screen_Time_Hours,
            Social_Media_Hours,
            Sleep_Hours,
            Study_Work_Hours,
            Physical_Activity_Min,
            App_Unlocks_Per_Day,
            Mood_Score
        ]])

        prediction = model.predict(features)[0]

        st.markdown("### 🎯 AI Assessment")

        if prediction == "Low":
            st.markdown("""
            <div style="padding:25px;border-radius:14px;background:#E8F5E9;text-align:center;">
                <h2 style="color:#2E7D32;">🟢 Low Addiction Risk</h2>
                <p>Balanced digital usage and healthy lifestyle indicators.</p>
            </div>
            """, unsafe_allow_html=True)

        elif prediction == "Medium":
            st.markdown("""
            <div style="padding:25px;border-radius:14px;background:#FFF8E1;text-align:center;">
                <h2 style="color:#F9A825;">🟡 Moderate Addiction Risk</h2>
                <p>Elevated digital engagement detected — monitor behavior.</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style="padding:25px;border-radius:14px;background:#FFEBEE;text-align:center;">
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
</div>
""", unsafe_allow_html=True)
