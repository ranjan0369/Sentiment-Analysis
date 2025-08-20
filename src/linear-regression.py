import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import numpy as np

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode labels
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

# Feature: word count
df['review_word_count'] = df['review'].apply(lambda x: len(str(x).split()))

X = df[['review_word_count']]  # independent variable
y = df['sentiment']            # dependent variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)

# Predictions
y_pred = lr.predict(X_test)

# Convert regression output to classification (threshold at 0.5)
y_pred_class = [1 if val >= 0.5 else 0 for val in y_pred]

# Metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))
print("Classification Accuracy (using 0.5 threshold):", accuracy_score(y_test, y_pred_class))