import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load trained model and columns
with open("model/salary_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Encoding maps (must match training)
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


# 💡 Prediction Function
def predict_salary(data: dict):
    data["Gender"] = gender_map.get(data["Gender"], 0)
    data["Education Level"] = education_map.get(data["Education Level"], 0)
    data["Job Title"] = job_title_map.get(data["Job Title"], 0)

    df = pd.DataFrame([data])

    for col in model_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_columns]
    prediction = model.predict(df)
    return float(prediction[0])


# 🧠 App UI
st.set_page_config(page_title="Employee Salary Predictor", layout="centered")

st.markdown("<h1 style='text-align:center; color:#4CAF50;'>💼 AI-Powered Salary Predictor</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center;'>Estimate employee salary based on profile</h5>", unsafe_allow_html=True)

# 🔽 Input Fields
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 65, 30)
    gender = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

with col2:
    job_title = st.selectbox("Job Title", [
        "Data Analyst", "Software Engineer", "Project Manager", "HR Specialist", "Data Scientist"
    ])
    experience = st.slider("Years of Experience", 0, 40, 5)

# 🔘 Predict Button
if st.button("🔮 Predict Salary"):
    input_data = {
        "Age": age,
        "Gender": gender,
        "Education Level": education,
        "Job Title": job_title,
        "Years of Experience": experience
    }

    salary = predict_salary(input_data)
    st.success(f"💰 Estimated Salary: ₹{salary:,.2f}")


# 📊 Load original data for charts
if st.checkbox("📈 Show Salary Data Insights"):
    try:
        df = pd.read_csv("Salary Data.csv")

        st.subheader("📌 Salary Distribution")
        fig1, ax1 = plt.subplots()
        sns.histplot(df["Salary"], kde=True, ax=ax1, color="skyblue")
        st.pyplot(fig1)

        st.subheader("📌 Salary vs Experience")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x="Education Level", y="Salary", data=df, ax=ax2)
        st.pyplot(fig2)

    except Exception as e:
        st.warning(f"Error loading Salary Data.csv: {e}")
