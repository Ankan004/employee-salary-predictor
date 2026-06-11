import pickle
import pandas as pd
import shap


with open("models/best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


def explain_prediction(input_data):

    X = pd.DataFrame([input_data])

    preprocessor = pipeline.named_steps["preprocessor"]

    model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(X)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(transformed)

    feature_names = preprocessor.get_feature_names_out()

    explanation = pd.DataFrame({
        "Feature": feature_names,
        "Contribution": shap_values[0]
    })

    explanation["Absolute"] = (
        explanation["Contribution"]
        .abs()
    )

    explanation = (
        explanation
        .sort_values(
            "Absolute",
            ascending=False
        )
        .head(10)
    )

    return explanation
