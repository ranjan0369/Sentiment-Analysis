from feature_extractor import vectorizer
from train_model import model
feature_names = vectorizer.get_feature_names_out()
coefficients = model.coef_[0]

top_positive = sorted(zip(coefficients, feature_names), reverse=True)[:20]
top_negative = sorted(zip(coefficients, feature_names))[:20]

print("Top Positive Words:", top_positive)
print("Top Negative Words:", top_negative)
