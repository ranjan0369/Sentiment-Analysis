from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

df = pd.read_csv("imdb_dataset_cleaned.csv")

vectorizer = TfidfVectorizer(max_features=5000)   # keep top 5000 words or features
X = vectorizer.fit_transform(df['clean_review'])
dfx = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
y = df['sentiment'].apply(lambda x: 1 if x == "positive" else 0)
print(f"Feature extraction complete. Shape of X: {X.shape}, Number of features: {len(vectorizer.get_feature_names_out())}")
print(f"{X.shape[0]} reviews processed with {len(vectorizer.get_feature_names_out())} features.")
# print(f"{dfx.head}")
# dfx.to_csv("features.csv", index=False)