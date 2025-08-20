from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Your 3 test reviews
reviews = [
    "The movie was fantastic, inspiring and mindblowing movie movie",
    "Worst movie of all time",
    "Fantastic direction and great acting. I would recommend others to watch the movie"
]

# Step 1: TF-IDF WITHOUT normalization
vectorizer_raw = TfidfVectorizer(stop_words='english', norm=None)  # norm=None disables normalization
X_raw = vectorizer_raw.fit_transform(reviews)
df_raw = pd.DataFrame(X_raw.toarray(), columns=vectorizer_raw.get_feature_names_out())
print("Unnormalized TF-IDF Matrix:\n")
print(df_raw)

# Step 2: TF-IDF WITH default normalization (L2)
vectorizer_norm = TfidfVectorizer(stop_words='english')  # default norm='l2'
X_norm = vectorizer_norm.fit_transform(reviews)
df_norm = pd.DataFrame(X_norm.toarray(), columns=vectorizer_norm.get_feature_names_out())
print("\nNormalized TF-IDF Matrix:\n")
print(df_norm)
