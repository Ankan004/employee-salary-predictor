import pandas as pd
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load dataset
df = pd.read_csv("data/Salary Data.csv")
df.dropna(inplace=True)

# Encode categorical columns
label_encoders = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature and target
X = df.drop("Salary", axis=1)
y = df["Salary"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models to train
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    results[name] = {
        "model": model,
        "R2": r2_score(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "RMSE": np.sqrt(mean_squared_error(y_test, preds))
    }

# Find best model
best_model_name = max(results, key=lambda name: results[name]["R2"])
best_model = results[best_model_name]["model"]

print(f"✅ Best model: {best_model_name}")
print(f"R²: {results[best_model_name]['R2']:.3f}")
print(f"MAE: {results[best_model_name]['MAE']:.2f}")
print(f"RMSE: {results[best_model_name]['RMSE']:.2f}")

# Save best model
os.makedirs("model", exist_ok=True)

with open("model/salary_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("model/model_columns.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
