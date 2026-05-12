# Lab 08 - Capstone Student Performance

Module: 8 - Mini Project

Dataset:

https://www.kaggle.com/datasets/nikhil7280/student-performance-multiple-linear-regression

Reference:

https://github.com/data-flair/machine-learning-projects

## Goal

Let learners complete a small end-to-end AI project.

## Project Problem

Predict student performance using available student-related features.

## Learner Task

1. Load the dataset.
2. Understand the columns.
3. Clean missing or duplicate data.
4. Select target variable.
5. Train a regression model.
6. Evaluate the model.
7. Explain the result in simple language.

## Simple Python Starter

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("dataset/Student_Performance.csv")

print(df.head())
print(df.info())
print(df.isnull().sum())

target_column = "Performance Index"

X = df.drop(target_column, axis=1)
y = df[target_column]

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, predictions))
print("R2 Score:", r2_score(y_test, predictions))
```

## Expected Output

Each learner or group should present:

- Problem statement.
- Dataset columns.
- Cleaning steps.
- Model used.
- Evaluation result.
- One improvement idea.

## Trainer Tip

For beginners, focus more on project flow than perfect accuracy.
