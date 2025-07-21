# 💼 Employee Salary Predictor

This is a machine learning web app that predicts an employee's salary based on their gender, education level, and years of experience.

🔗 **Live App**: [https://ankan004.streamlit.app](https://ankan004.streamlit.app)

---

## ✨ Features

- ✅ Predicts salaries using a trained ML model
- ✅ Built with Streamlit for a smooth user experience
- ✅ Easy and clean UI
- ✅ Includes graphs and performance metrics

---

## ⚙️ Technologies Used

- Python 3
- Pandas & NumPy
- Scikit-learn
- Streamlit
- Matplotlib / Seaborn

---

## 📁 Project Structure

employee-salary-predictor/
├── app.py # Main Streamlit app
├── train.py # Model training script
├── predict.py # Prediction script
├── model/
│ ├── salary_model.pkl # Trained Random Forest model
│ └── model_columns.pkl # Feature columns for prediction
├── data/
│ └── Salary Data.csv # Employee salary dataset
├── requirements.txt # Required Python packages
└── README.md # Project overview (this file)


---

# 📦 Step 1: Clone the repository
git clone https://github.com/Ankan004/employee-salary-predictor.git
cd employee-salary-predictor

# 🐍 Step 2: (Optional) Create a virtual environment
python -m venv venv
venv\Scripts\activate         # On Windows
# source venv/bin/activate   # On macOS/Linux

# 📚 Step 3: Install dependencies
pip install -r requirements.txt

# 🚀 Step 4: Run the Streamlit app
streamlit run app.py
------

🤖 Model Performance
The Employee Salary Prediction model is built using a Random Forest Regressor, selected after evaluating multiple algorithms for accuracy and consistency. It achieved an impressive R² score of 0.940, indicating that 94% of the variance in salary data is explained by the model. The Mean Absolute Error (MAE) is ₹8,524.79, and the Root Mean Squared Error (RMSE) is ₹11,981.83, demonstrating the model's strong predictive capability and robustness in estimating employee salaries based on input features.

🙋‍♂️ Author
Developed with ❤️ by Ankan Ghosh
📬 Feel free to connect or contribute!



