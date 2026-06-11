import pandas as pd

df = pd.read_csv("data/Salary_Data.csv")

average_salary = df["Salary"].mean()

print("Average Salary")
print(round(average_salary, 2))