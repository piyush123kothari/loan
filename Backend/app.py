from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

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
        joblib.load("model/adaboost.pkl"),

    "XGBoost":
        joblib.load("model/xgboost.pkl")
}

@app.route("/")
def home():
    return jsonify({"message": "Loan Prediction API Running"})

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    selected_model = data["model_name"]

    model = MODELS[selected_model]

    input_df = pd.DataFrame([{
        "no_of_dependents": data["no_of_dependents"],
        "education": data["education"],
        "self_employed": data["self_employed"],
        "income_annum": data["income_annum"],
        "loan_amount": data["loan_amount"],
        "loan_term": data["loan_term"],
        "cibil_score": data["cibil_score"],
        "residential_assets_value": data["residential_assets_value"],
        "commercial_assets_value": data["commercial_assets_value"],
        "luxury_assets_value": data["luxury_assets_value"],
        "bank_asset_value": data["bank_asset_value"],
        "exam_qualified": data["exam_qualified"],
        "admission_type": data["admission_type"]
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0]

    result = {
        "loan_status": "Approved" if prediction == 1 else "Rejected",
        "confidence": round(max(probability) * 100, 2)
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)