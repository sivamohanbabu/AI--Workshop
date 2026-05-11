# Lab 04 - Predictive Modeling Regression

Module: 5 - Core ML Algorithms

Dataset:

https://www.kaggle.com/datasets/zafarali27/house-price-prediction-dataset

Reference:

https://github.com/arpitaapatel/Iris-Classification

## Goal

Teach learners how to build a simple regression model to predict a number.

## Trainer Explanation

Regression is used when the output is a numeric value.

Examples:

- Predict house price.
- Predict salary.
- Predict sales.
- Predict marks.

## Learner Task

1. Load the house price dataset.
2. Select input columns and target column.
3. Split data into training and testing sets.
4. Train a Linear Regression model.
5. Predict on test data.
6. Check model error.

## Simple Python Starter

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("dataset/House Price Prediction Dataset.csv")

print(df.head())
print(df.columns)

X = df.drop(["Id", "Price"], axis=1)
y = df["Price"]

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

Learners should understand:

- Input variables are called features.
- Output variable is called target.
- Regression predicts numbers.
- Error tells us how far predictions are from actual values.

## Trainer Tip

If column names are different in the downloaded dataset, show learners how to use `df.columns` and update the code.
