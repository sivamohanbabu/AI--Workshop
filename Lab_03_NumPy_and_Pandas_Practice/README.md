# Lab 03 - NumPy and Pandas Practice

Module: 4 - Python for AI

Dataset:

https://www.kaggle.com/datasets/abhishek14398/salary-dataset-simple-linear-regression

Reference:

https://github.com/0xKatie/ai-ml-mastery-hub

## Goal

Help learners practice basic Python data operations used in AI projects.

## Trainer Explanation

NumPy is used for numerical operations. Pandas is used for tabular data.

Learners should understand:

- Series and DataFrame.
- Selecting columns.
- Filtering rows.
- Mean, max, min.
- Creating new columns.

## Learner Task

1. Load the salary dataset.
2. Show first 5 rows.
3. Find average salary.
4. Find minimum and maximum years of experience.
5. Create a new column called `Salary_Level`.
6. Filter employees with more than 5 years experience.

## Simple Python Starter

```python
import pandas as pd
import numpy as np

df = pd.read_csv("dataset/Salary_dataset.csv")

print(df.head())
print("Average salary:", df["Salary"].mean())
print("Min experience:", df["YearsExperience"].min())
print("Max experience:", df["YearsExperience"].max())

df["Salary_Level"] = np.where(df["Salary"] > df["Salary"].mean(), "High", "Low")

print(df[df["YearsExperience"] > 5])
print(df.head())
```

## Expected Output

Learners should understand how to manipulate a dataset using Pandas and NumPy.

## Trainer Tip

Tell learners that Pandas is like Excel for Python, but more powerful for automation and AI.
