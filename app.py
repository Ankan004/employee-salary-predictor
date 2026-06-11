
import streamlit as st
import pandas as pd
import pickle
import json
import plotly.express as px

from src.report_generator import generate_report
from src.utils import career_level
from src.shap_explainer import explain_prediction

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="AI Salary Analytics",
    page_icon="💼",
    layout="wide"
)

# =====================================
# LOAD DATA
# =====================================

dataset = pd.read_csv("data/Salary_Data.csv")

job_titles = sorted(
    dataset["Job Title"]
    .dropna()
    .unique()
)

# =====================================
# LOAD MODEL
# =====================================

with open("models/best_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# =====================================
# LOAD METRICS
# =====================================

with open("models/metrics.json", "r") as f:
    metrics = json.load(f)

# =====================================
# FUNCTIONS
# =====================================

def predict_salary(data):

    df = pd.DataFrame([data])

    prediction = pipeline.predict(df)

    return float(prediction[0])

# =====================================
# HEADER
# =====================================

st.markdown("""
# 💼 AI-Powered Employee Salary Analytics & Prediction System

Predict employee salaries using Machine Learning models including
**Linear Regression, Random Forest, and XGBoost**.
""")

# =====================================
# TABS
# =====================================

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔮 Prediction",
        "📊 Analytics",
        "📈 Model Performance",
        "🔍 Feature Importance",
        "🧠 Explainable AI"
    ]
)


# =====================================
# TAB 1 : PREDICTION
# =====================================

with tab1:

    st.subheader("Employee Salary Prediction")

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider(
            "Age",
            18,
            65,
            30
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        education = st.selectbox(
            "Education Level",
            [
                "High School",
                "Bachelor's",
                "Master's",
                "PhD"
            ]
        )

    with col2:

        job_title = st.selectbox(
            "Job Title",
            job_titles
        )

        experience = st.slider(
            "Years of Experience",
            0,
            40,
            5
        )

    if st.button("🚀 Predict Salary"):

        input_data = {
            "Age": age,
            "Gender": gender,
            "Education Level": education,
            "Job Title": job_title,
            "Years of Experience": experience
        }

        salary = predict_salary(input_data)
        shap_result = explain_prediction(input_data)

        st.session_state["shap_result"] = shap_result

        level = career_level(salary)

        avg_salary = dataset["Salary"].mean()

        percentile = (
            (dataset["Salary"] < salary)
            .mean()
            * 100
        )

        difference = salary - avg_salary

        # ===========================
        # KPI CARDS
        # ===========================

        kpi1, kpi2, kpi3 = st.columns(3)

        with kpi1:
            st.metric(
                "💰 Predicted Salary",
                f"₹{salary:,.0f}"
            )

        with kpi2:
            st.metric(
                "📊 Salary Percentile",
                f"{percentile:.1f}%"
            )

        with kpi3:
            st.metric(
                "📈 Dataset Average",
                f"₹{avg_salary:,.0f}"
            )

        st.divider()

        # ===========================
        # BENCHMARK
        # ===========================

        if difference > 0:

            st.success(
                f"🚀 Estimated salary is ₹{difference:,.0f} above the dataset average."
            )

        else:

            st.warning(
                f"📉 Estimated salary is ₹{abs(difference):,.0f} below the dataset average."
            )

        # ===========================
        # PERCENTILE INSIGHT
        # ===========================

        if percentile >= 90:

            st.success(
                "🏆 Top 10% salary bracket"
            )

        elif percentile >= 75:

            st.success(
                "📈 Above-average earning potential"
            )

        elif percentile >= 50:

            st.info(
                "📊 Around the median salary range"
            )

        else:

            st.warning(
                "📉 Below the median salary range"
            )

        # ===========================
        # CAREER INSIGHT
        # ===========================

        st.info(
            f"""
            **Career Level:** {level}

            Based on your age, education level, job role and professional experience,
            the model predicts an annual salary of approximately **₹{salary:,.0f}**.

            Your predicted compensation lies in the
            **{percentile:.1f}th percentile** of the dataset.
            """
        )

        # ===========================
        # REPORT DOWNLOAD
        # ===========================

        report = generate_report(
            age,
            gender,
            education,
            job_title,
            experience,
            salary,
            level
        )

        csv = report.to_csv(index=False)

        st.download_button(
            label="📥 Download Prediction Report",
            data=csv,
            file_name="salary_prediction_report.csv",
            mime="text/csv"
        )





# =====================================
# TAB 2 : ANALYTICS
# =====================================

with tab2:

    st.subheader("Salary Dataset Analytics")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Records",
        len(dataset)
    )

    c2.metric(
        "Features",
        len(dataset.columns) - 1
    )

    c3.metric(
        "Target",
        "Salary"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        dataset.head(10),
        use_container_width=True
    )

    st.divider()

    fig1 = px.histogram(
        dataset,
        x="Salary",
        nbins=30,
        title="Salary Distribution"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    fig2 = px.scatter(
        dataset,
        x="Years of Experience",
        y="Salary",
        color="Education Level",
        title="Experience vs Salary"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    fig3 = px.box(
        dataset,
        x="Education Level",
        y="Salary",
        title="Education Level vs Salary"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    fig4 = px.pie(
        dataset,
        names="Gender",
        title="Gender Distribution"
    )

    st.plotly_chart(
        fig4,
        use_container_width=True
    )

# =====================================
# TAB 3 : MODEL PERFORMANCE
# =====================================

with tab3:

    st.subheader("Model Evaluation")

    best_model = metrics["best_model"]

    result = metrics["results"][best_model]

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "R² Score",
        f"{result['R2']:.3f}"
    )

    c2.metric(
        "MAE",
        f"{result['MAE']:.0f}"
    )

    c3.metric(
        "RMSE",
        f"{result['RMSE']:.0f}"
    )

    st.success(
        f"🏆 Best Model: {best_model}"
    )

    st.divider()

    comparison_df = pd.DataFrame(
        metrics["results"]
    ).T

    st.subheader("Model Comparison")

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    fig = px.bar(
        comparison_df.reset_index(),
        x="index",
        y="R2",
        title="R² Score Comparison",
        labels={
            "index": "Model",
            "R2": "R² Score"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TAB 4 : EXPLAINABILITY
# =====================================

with tab4:

    st.subheader(
        "Feature Importance Analysis"
    )

    feature_df = pd.read_csv(
        "models/feature_importance.csv"
    )

    st.dataframe(
        feature_df.head(20),
        use_container_width=True
    )

    fig = px.bar(
        feature_df.head(15),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top 15 Most Important Features"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =====================================
# TAB 5 : SHAP EXPLAINABILITY
# =====================================

with tab5:

    st.subheader("🧠 Explainable AI (SHAP)")

    st.markdown("""
    SHAP (SHapley Additive Explanations) helps explain
    why the model predicted a particular salary.
    """)

    if "shap_result" not in st.session_state:

        st.info(
            "Make a prediction first to see SHAP explanations."
        )

    else:

        shap_df = st.session_state["shap_result"]

        st.dataframe(
            shap_df,
            use_container_width=True
        )

        fig = px.bar(
            shap_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            title="Top Contributors To This Prediction"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        positive = shap_df[
            shap_df["Contribution"] > 0
        ].head(3)

        negative = shap_df[
            shap_df["Contribution"] < 0
        ].head(3)

        st.subheader("📈 Positive Contributors")

        st.dataframe(
            positive,
            use_container_width=True
        )

        st.subheader("📉 Negative Contributors")

        st.dataframe(
            negative,
            use_container_width=True
        )

