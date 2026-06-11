import pickle
import pandas as pd

with open("models/best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)


def predict_salary(input_data):

    df = pd.DataFrame([input_data])

    prediction = pipeline.predict(df)

    return float(prediction[0])


if __name__ == "__main__":

    sample = {
        "Age": 28,
        "Gender": "Male",
        "Education Level": "Master's",
        "Job Title": "Data Analyst",
        "Years of Experience": 5
    }

    salary = predict_salary(sample)

    print(f"Predicted Salary: ₹{salary:,.2f}")