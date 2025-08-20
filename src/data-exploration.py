# 📌 Sentiment Analysis Dataset Exploration (IMDB)
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load Dataset
df = pd.read_csv("dataset.csv")

# 2. Inspect Dataset
print("First 5 rows:")
print(df.head())
print("\nDataset Shape (rows, columns):", df.shape)

# 3. Check Features & Labels
print("\nColumns:", df.columns.tolist())
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# 4. Data Types & Null Values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Basic Statistics
print("\nSentiment Distribution (normalized):")
print(df['sentiment'].value_counts(normalize=True))

# 6. Explore Text Features
df['review_length'] = df['review'].apply(len)
df['review_word_count'] = df['review'].apply(lambda x: len(x.split()))

print("\nReview Length & Word Count Stats:")
print(df[['review_length', 'review_word_count']].describe())

# Show some examples
print("\nExample Review:")
print(df['review'][0])
print("Sentiment:", df['sentiment'][0])

# 7. Visualization
plt.figure(figsize=(12,5))

# Histogram of word counts
plt.subplot(1,2,1)
df['review_word_count'].hist(bins=50, color='skyblue')
plt.xlabel("Word Count")
plt.ylabel("Number of Reviews")
plt.title("Distribution of Review Word Counts")

# Bar chart of sentiment distribution
plt.subplot(1,2,2)
df['sentiment'].value_counts().plot(kind='bar', color=['orange','green'])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")

plt.tight_layout()
plt.show()

# 8. Clean Labels for Modeling (convert to 0/1)
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

print("\nAfter Label Encoding:")
print(df.head())