import pickle
import pandas as pd

with open("models/best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

model = pipeline.named_steps["model"]

preprocessor = pipeline.named_steps["preprocessor"]

feature_names = preprocessor.get_feature_names_out()

importance = model.feature_importances_

feature_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=False
)

feature_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print(feature_df.head(20))
print("\nSaved:")
print("models/feature_importance.csv")