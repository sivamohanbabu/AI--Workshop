# Lab 02 - EDA and Data Cleaning

Module: 3 - Data Preprocessing

Dataset:

https://www.kaggle.com/datasets/ashydv/advertising-dataset

Reference:

https://github.com/ZainUlAbideen02/Learn-Numpy-in-4-Easy-Phases

## Goal

Teach learners how to inspect a dataset, find basic issues, and clean it before modeling.

## Trainer Explanation

EDA means Exploratory Data Analysis. Before building any AI model, we must understand the data.

Main checks:

- Number of rows and columns.
- Column names.
- Missing values.
- Duplicate rows.
- Data types.
- Basic statistics.
- Simple charts.

## Learner Task

1. Load the advertising dataset.
2. Display the first 5 rows.
3. Check missing values.
4. Check duplicate rows.
5. Show summary statistics.
6. Plot one simple chart.
7. Write 3 observations about the data.

## Simple Python Starter

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("dataset/advertising.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())
print(df.describe())

df.hist(figsize=(8, 6))
plt.tight_layout()
plt.show()
```

## Expected Output

Learners should be able to say:

- Which columns exist.
- Whether missing values are present.
- Whether duplicates are present.
- Which variables look important.

## Trainer Tip

Explain that most AI project time is spent on data understanding and cleaning, not only model training.
