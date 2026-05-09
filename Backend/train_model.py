import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

# ======================================
# Load Dataset
# ======================================

df = pd.read_csv("loan_status_prediction_dataset_50000_with_exam_feature.csv")

# ======================================
# Drop loan_id column
# ======================================

df.drop(columns=["loan_id"], inplace=True)

# ======================================
# Encode Target Variable
# ======================================

df["loan_status"] = df["loan_status"].map({
    "Approved": 1,
    "Rejected": 0
})

# ======================================
# Features and Target
# ======================================

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# ======================================
# Column Types
# ======================================

categorical_cols = [
    "education",
    "self_employed",
    "exam_qualified",
    "admission_type"
]

numerical_cols = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value"
]

# ======================================
# Preprocessing
# ======================================

preprocessor = ColumnTransformer([
    (
        "cat",
        OneHotEncoder(drop="first"),
        categorical_cols
    ),
    (
        "num",
        StandardScaler(),
        numerical_cols
    )
])

# ======================================
# Models
# ======================================

models = {

    "logistic_regression.pkl":
        LogisticRegression(max_iter=1000),

    "random_forest.pkl":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        ),

    "gradient_boosting.pkl":
        GradientBoostingClassifier(),

    "decision_tree.pkl":
        DecisionTreeClassifier(),

    "adaboost.pkl":
        AdaBoostClassifier(),

    "xgboost.pkl":
        XGBClassifier(
            eval_metric='logloss'
        )
}

# ======================================
# Train Test Split
# ======================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================
# Create model folder
# ======================================

os.makedirs("model", exist_ok=True)

# ======================================
# Train and Save Models
# ======================================

for filename, model in models.items():

    print("\n" + "=" * 60)

    print(f"Training Model: {filename}")

    print("=" * 60)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(y_test, y_pred)

    recall = recall_score(y_test, y_pred)

    f1 = f1_score(y_test, y_pred)

    # Save model
    joblib.dump(
        pipeline,
        f"model/{filename}"
    )

    # Print Results
    print(f"Model Saved: {filename}")

    print(f"Accuracy : {accuracy:.4f}")

    print(f"Precision: {precision:.4f}")

    print(f"Recall   : {recall:.4f}")

    print(f"F1 Score : {f1:.4f}")

    print("\nClassification Report:\n")

    print(classification_report(y_test, y_pred))

    print("-" * 60)

print("\nAll models trained successfully.")