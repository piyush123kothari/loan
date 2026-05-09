from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os

app = Flask(__name__)

CORS(app)

# ======================================
# Load Models
# ======================================

MODELS = {

    "Logistic Regression":
        joblib.load("model/logistic_regression.pkl"),

    "Random Forest":
        joblib.load("model/random_forest.pkl"),

    "Gradient Boosting":
        joblib.load("model/gradient_boosting.pkl"),

    "Decision Tree":
        joblib.load("model/decision_tree.pkl"),

    "AdaBoost":
        joblib.load("model/adaboost.pkl")
}

# ======================================
# Home Route
# ======================================

@app.route("/")
def home():
    return "Loan Prediction API Running Successfully"

# ======================================
# Prediction Route
# ======================================

@app.route("/predict", methods=["POST"])

def predict():

    data = request.json

    model_name = data["model_name"]

    model = MODELS[model_name]

    input_data = pd.DataFrame([{

        "no_of_dependents":
            int(data["no_of_dependents"]),

        "education":
            data["education"],

        "self_employed":
            data["self_employed"],

        "income_annum":
            float(data["income_annum"]),

        "loan_amount":
            float(data["loan_amount"]),

        "loan_term":
            float(data["loan_term"]),

        "cibil_score":
            float(data["cibil_score"]),

        "residential_assets_value":
            float(data["residential_assets_value"]),

        "commercial_assets_value":
            float(data["commercial_assets_value"]),

        "luxury_assets_value":
            float(data["luxury_assets_value"]),

        "bank_asset_value":
            float(data["bank_asset_value"]),

        "exam_qualified":
            data["exam_qualified"],

        "admission_type":
            data["admission_type"]
    }])

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0]

    confidence = round(max(probability) * 100, 2)

    result = "Approved" if prediction == 1 else "Rejected"

    return jsonify({
        "prediction": result,
        "confidence": confidence
    })

# ======================================
# Run App
# ======================================

if __name__ == "__main__":
    app.run(debug=True)