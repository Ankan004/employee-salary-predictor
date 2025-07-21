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
Metric	                                Value
Best Model	                       Random Forest Regressor
R² Score                              	0.940
MAE                                 	₹8,524.79
RMSE                                	₹11,981.83

🙋‍♂️ Author
Developed with ❤️ by Ankan Ghosh
📬 Feel free to connect or contribute!



