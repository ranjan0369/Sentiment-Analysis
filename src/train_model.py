
from feature_extractor import vectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from feature_extractor import X, y
import joblib

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
joblib.dump(model, "models/logreg_imdb_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")

#model evaluation
y_pred = model.predict(X_test)

print("Accuracy:", (accuracy_score(y_test, y_pred))*100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
# cm = confusion_matrix(y_test, y_pred)
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# feature_names = vectorizer.get_feature_names_out()
# coefficients = model.coef_[0]

# top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:20]
# top_negative = sorted(zip(coefficients, feature_names))[:20]

# print("Top Positive Words:", top_positive)
# print("Top Negative Words:", top_negative)


