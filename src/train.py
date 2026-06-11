import pandas as pd
import numpy as np
import pickle
import json
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("data/Salary_Data.csv")
df.dropna(inplace=True)

# -------------------------
# Features & Target
# -------------------------
X = df.drop("Salary", axis=1)
y = df["Salary"]

# -------------------------
# Column Types
# -------------------------
categorical_features = [
    "Gender",
    "Education Level",
    "Job Title"
]

numerical_features = [
    "Age",
    "Years of Experience"
]

# -------------------------
# Preprocessor
# -------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            "passthrough",
            numerical_features
        )
    ]
)

# -------------------------
# Train Test Split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -------------------------
# Models
# -------------------------
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42
    )
}

results = {}

best_pipeline = None
best_score = -999

# -------------------------
# Training Loop
# -------------------------
for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    results[model_name] = {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse)
    }

    print(f"\n{model_name}")
    print(f"R²   : {r2:.4f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    if r2 > best_score:
        best_score = r2
        best_pipeline = pipeline
        best_model_name = model_name

# -------------------------
# Save Models
# -------------------------
os.makedirs("models", exist_ok=True)

with open("models/best_pipeline.pkl", "wb") as f:
    pickle.dump(best_pipeline, f)

# -------------------------
# Save Metrics
# -------------------------
metrics = {
    "best_model": best_model_name,
    "results": results
}

with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("\n====================")
print("BEST MODEL")
print("====================")
print(best_model_name)
print(f"R² = {best_score:.4f}")

print("\nSaved:")
print("models/best_pipeline.pkl")
print("models/metrics.json")