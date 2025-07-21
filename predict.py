import pickle
import pandas as pd

# Load trained model
with open("model/salary_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load model columns
with open("model/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Define encoders (same mapping used in training)
gender_map = {"Male": 1, "Female": 0}
education_map = {
    "High School": 0,
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}
job_title_map = {
    "Data Analyst": 0,
    "Software Engineer": 1,
    "Project Manager": 2,
    "HR Specialist": 3,
    "Data Scientist": 4
}

def preprocess_input(data):
    data["Gender"] = gender_map.get(data.get("Gender", ""), 0)
    data["Education Level"] = education_map.get(data.get("Education Level", ""), 0)
    data["Job Title"] = job_title_map.get(data.get("Job Title", ""), 0)
    return data

def predict_salary(input_data: dict):
    """
    Predict salary using saved model
    :param input_data: dict with raw input (e.g., strings)
    :return: predicted salary
    """
    input_data = preprocess_input(input_data)
    df = pd.DataFrame([input_data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    prediction = model.predict(df)
    return float(prediction[0])

# Example usage
if __name__ == "__main__":
    example_input = {
        "Age": 28,
        "Gender": "Male",
        "Education Level": "Master's",
        "Job Title": "Data Analyst",
        "Years of Experience": 5
    }

    predicted = predict_salary(example_input)
    print(f"Predicted Salary: ₹{predicted:,.2f}")
