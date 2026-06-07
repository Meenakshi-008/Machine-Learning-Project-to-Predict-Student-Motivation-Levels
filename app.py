from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load model artifacts if they exist
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
COLUMNS_PATH = "columns.pkl"

model, scaler, columns = None, None, None
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    columns = joblib.load(COLUMNS_PATH)
except:
    print("Model files not found. Running in demo mode.")

LABEL_MAP = {0: "High", 1: "Low", 2: "Medium"}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        age = float(data.get("age", 17))
        study_time = float(data.get("study_time", 10))
        absences = float(data.get("absences", 3))
        tutoring = float(data.get("tutoring", 0))
        parental_support = float(data.get("parental_support", 2))
        extracurricular = float(data.get("extracurricular", 0))
        sports = float(data.get("sports", 0))
        music = float(data.get("music", 0))
        volunteering = float(data.get("volunteering", 0))

        study_efficiency = study_time / (absences + 1)
        support_score = parental_support + tutoring
        activity_score = sports + music + volunteering

        features = np.array([[age, study_time, absences, tutoring, parental_support,
                               extracurricular, sports, music, volunteering,
                               study_efficiency, support_score, activity_score]])

        if model and scaler:
            features_scaled = scaler.transform(features)
            pred = model.predict(features_scaled)[0]
            proba = model.predict_proba(features_scaled)[0].tolist()
            label = LABEL_MAP.get(int(pred), "Medium")
        else:
            # Demo mode: simple heuristic
            score = (study_time * 0.4 + parental_support * 0.2 +
                     activity_score * 0.1 - absences * 0.3 + support_score * 0.1)
            if score > 5:
                label, proba = "High", [0.1, 0.15, 0.75]
            elif score > 2:
                label, proba = "Medium", [0.15, 0.65, 0.2]
            else:
                label, proba = "Low", [0.7, 0.2, 0.1]

        return jsonify({
            "prediction": label,
            "probabilities": {
                "Low": round(proba[1] if model else proba[0], 3),
                "Medium": round(proba[2] if model else proba[1], 3),
                "High": round(proba[0] if model else proba[2], 3)
            },
            "inputs": {
                "study_efficiency": round(study_efficiency, 2),
                "support_score": round(support_score, 2),
                "activity_score": round(activity_score, 2)
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/analytics")
def analytics():
    """Return mock analytics data for charts"""
    import random
    random.seed(42)

    # Simulate dataset distributions
    motivation_dist = {"Low": 312, "Medium": 478, "High": 210}

    study_time_hist = {
        "bins": ["0-5h", "5-10h", "10-15h", "15-20h", "20-25h", "25-30h"],
        "counts": [45, 132, 210, 178, 95, 40]
    }

    study_vs_motivation = {
        "Low": {"min": 0, "q1": 3.2, "median": 6.1, "q3": 9.8, "max": 18},
        "Medium": {"min": 2, "q1": 7.5, "median": 12.3, "q3": 17.1, "max": 28},
        "High": {"min": 8, "q1": 14.2, "median": 19.5, "q3": 24.8, "max": 30}
    }

    feature_importance = {
        "Absences": 0.22,
        "StudyTimeWeekly": 0.19,
        "study_efficiency": 0.16,
        "ParentalSupport": 0.12,
        "support_score": 0.09,
        "Tutoring": 0.07,
        "activity_score": 0.05,
        "Age": 0.04,
        "Extracurricular": 0.03,
        "Sports": 0.01,
        "Music": 0.01,
        "Volunteering": 0.01
    }

    correlation_matrix = {
        "labels": ["GPA", "StudyTime", "Absences", "Support", "Tutoring", "Activity"],
        "data": [
            [1.0, 0.72, -0.65, 0.48, 0.41, 0.23],
            [0.72, 1.0, -0.58, 0.36, 0.45, 0.31],
            [-0.65, -0.58, 1.0, -0.29, -0.22, -0.15],
            [0.48, 0.36, -0.29, 1.0, 0.52, 0.18],
            [0.41, 0.45, -0.22, 0.52, 1.0, 0.12],
            [0.23, 0.31, -0.15, 0.18, 0.12, 1.0]
        ]
    }

    age_distribution = {
        "ages": [15, 16, 17, 18],
        "counts": [198, 312, 264, 226]
    }

    return jsonify({
        "motivation_dist": motivation_dist,
        "study_time_hist": study_time_hist,
        "study_vs_motivation": study_vs_motivation,
        "feature_importance": feature_importance,
        "correlation_matrix": correlation_matrix,
        "age_distribution": age_distribution,
        "model_accuracy": 89.4,
        "total_students": 1000,
        "avg_study_time": 14.3
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)