# 💼 AI-Powered Employee Salary Analytics & Prediction System

An advanced Machine Learning and Explainable AI application that predicts employee salaries based on demographic, educational, and professional attributes.

Built using Python, Streamlit, Scikit-Learn, XGBoost, Plotly, and SHAP.

---

## 🚀 Live Features

### 🔮 Salary Prediction
- Predict employee salary using trained ML models
- Interactive user inputs
- Real-time prediction results

### 📊 Analytics Dashboard
- Salary distribution analysis
- Experience vs Salary visualization
- Education vs Salary insights
- Gender distribution analysis
- Dataset preview and statistics

### 📈 Model Performance Comparison
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- Automatic best model selection

### 🔍 Feature Importance Analysis
- Global feature importance visualization
- Top influencing salary factors
- XGBoost feature ranking

### 🧠 Explainable AI (SHAP)
- Local prediction explanations
- Positive contributors
- Negative contributors
- SHAP-based interpretability

### 📄 Report Generation
- Download prediction reports as CSV
- Export salary insights

---

# 🏗️ Project Architecture

```text
employee-salary-predictor/
│
├── data/
│   └── Salary_Data.csv
│
├── models/
│   ├── best_pipeline.pkl
│   ├── metrics.json
│   └── feature_importance.csv
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── feature_importance.py
│   ├── report_generator.py
│   ├── benchmark.py
│   └── utils.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 🤖 Machine Learning Workflow

## Data Preprocessing

- Missing value handling
- Categorical encoding
- Pipeline-based preprocessing
- Train-Test Split

## Models Evaluated

| Model | Purpose |
|---------|---------|
| Linear Regression | Baseline Regression |
| Random Forest | Ensemble Learning |
| XGBoost | Gradient Boosting |

## Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

---

# 📈 Best Model Performance

| Metric | Score |
|----------|----------|
| Best Model | XGBoost |
| R² Score | 0.870 |
| MAE | 10,575 |
| RMSE | 17,626 |

---

# 🔍 Key Insights

The model identified the following features as most influential:

1. Age
2. Years of Experience
3. Education Level
4. Job Title
5. Gender

---

# 🛠️ Tech Stack

### Machine Learning
- Scikit-Learn
- XGBoost
- SHAP

### Data Analysis
- Pandas
- NumPy

### Visualization
- Plotly

### Frontend
- Streamlit

### Model Serialization
- Pickle

---

# 📸 Screenshots

## Prediction Dashboard

(Add Screenshot)

## Analytics Dashboard

(Add Screenshot)

## Model Performance

(Add Screenshot)

## Feature Importance

(Add Screenshot)

## Explainable AI (SHAP)

(Add Screenshot)

---

# ⚙️ Installation

Clone repository

```bash
git clone https://github.com/Ankan004/employee-salary-predictor.git
```

Move into project

```bash
cd employee-salary-predictor
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run application

```bash
streamlit run app.py
```

---

# 🎯 Future Improvements

- Salary forecasting
- Deep Learning models
- Automated report PDFs
- Cloud deployment
- Advanced Explainable AI visualizations

---

# 👨‍💻 Author

**Ankan Ghosh**

Full Stack Developer | AI Enthusiast | CSE Undergraduate

GitHub:
https://github.com/Ankan004

LinkedIn:
https://www.linkedin.com/in/ankan-ghosh-7a3b77335

---

⭐ If you found this project useful, consider giving it a star.