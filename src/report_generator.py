import pandas as pd

def generate_report(
    age,
    gender,
    education,
    job_title,
    experience,
    salary,
    level
):

    report = pd.DataFrame(
        {
            "Age": [age],
            "Gender": [gender],
            "Education": [education],
            "Job Title": [job_title],
            "Experience": [experience],
            "Predicted Salary": [salary],
            "Career Level": [level]
        }
    )

    return report