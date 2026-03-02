## FLASK
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "rf.pkl")
model = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        Screen_Time_Hours = float(request.form["screen_time"])
        Social_Media_Hours = float(request.form["social_media"])
        Sleep_Hours = float(request.form["sleep"])
        Physical_Activity_Min = float(request.form["activity"])
        App_Unlocks_Per_Day = float(request.form["unlocks"])

        features = np.array([[ 
            Screen_Time_Hours,
            Social_Media_Hours,
            Sleep_Hours,
            Physical_Activity_Min,
            App_Unlocks_Per_Day
        ]])

        prediction = model.predict(features)[0]

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":

    app.run(debug=True)
