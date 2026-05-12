# Lab 05 - Classification Lab

Module: 5 - Core ML Algorithms

Dataset:

https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Reference:

https://github.com/data-flair/machine-learning-projects

## Goal

Teach learners how to build a classification model to predict a category.

## Trainer Explanation

Classification is used when the output is a class or label.

Examples:

- Spam or not spam.
- Disease or no disease.
- Pass or fail.
- Fraud or normal transaction.

## Learner Task

1. Load the diabetes dataset.
2. Separate features and target.
3. Split the data.
4. Train a Logistic Regression model.
5. Predict test results.
6. Check accuracy and confusion matrix.

## Simple Python Starter

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("dataset/diabetes.csv")

print(df.head())
print(df.columns)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
```

## Expected Output

Learners should understand:

- Classification predicts categories.
- Accuracy shows correct predictions.
- Confusion matrix shows correct and incorrect class predictions.

## Trainer Tip

Explain false positive and false negative using simple healthcare examples.
