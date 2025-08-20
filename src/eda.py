# 📌 IMDB Sentiment Analysis - Full EDA

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# from collections import Counter
# from wordcloud import WordCloud
# import nltk
# from nltk.corpus import stopwords
# import re

# # Download stopwords if not already
# nltk.download('stopwords')
# stop_words = set(stopwords.words('english'))

# 1️⃣ Load Dataset
df = pd.read_csv("dataset.csv")

# 2️⃣ Basic Overview
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())
print(df.info())
print(df.isnull().sum())
print("\nSentiment distribution:\n", df['sentiment'].value_counts())

# 3️⃣ Label Encoding
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print("\nAfter encoding:\n", df['sentiment'].value_counts())

# 4️⃣ Feature Engineering
df['review_char_count'] = df['review'].apply(lambda x: len(str(x)))
df['review_word_count'] = df['review'].apply(lambda x: len(str(x).split()))
df['avg_word_length'] = df['review_char_count'] / df['review_word_count']
df['num_sentences'] = df['review'].apply(lambda x: str(x).count('.'))

# 5️⃣ Basic Statistics
print("\nReview Length & Word Count Stats:")
print(df[['review_char_count','review_word_count','avg_word_length','num_sentences']].describe())

# 6️⃣ Visualizations
plt.figure(figsize=(16,6))

# Word count distribution
plt.subplot(1,3,1)
sns.histplot(df['review_word_count'], bins=50, kde=True, color='skyblue')
plt.xlabel("Word Count")
plt.title("Word Count Distribution")

# Character count distribution
plt.subplot(1,3,2)
sns.histplot(df['review_char_count'], bins=50, kde=True, color='salmon')
plt.xlabel("Character Count")
plt.title("Character Count Distribution")

# Average word length
plt.subplot(1,3,3)
sns.histplot(df['avg_word_length'], bins=30, kde=True, color='green')
plt.xlabel("Average Word Length")
plt.title("Average Word Length Distribution")

plt.tight_layout()
plt.show()

# Correlation heatmap
plt.figure(figsize=(6,5))
sns.heatmap(df[['review_char_count','review_word_count','avg_word_length','num_sentences']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation between Text Features")
plt.show()

# 7️⃣ Text Content Analysis - Most Frequent Words
# def preprocess_text(text):
#     text = str(text).lower()  # lowercase
#     text = re.sub(r'<.*?>','', text)  # remove HTML tags
#     text = re.sub(r'[^a-zA-Z\s]','', text)  # remove special characters
#     words = text.split()
#     words = [w for w in words if w not in stop_words]
#     return words

# all_words = []
# for review in df['review']:
#     all_words.extend(preprocess_text(review))

# word_freq = Counter(all_words)
# most_common_words = word_freq.most_common(20)
# print("\n20 Most Common Words:\n", most_common_words)

# # 8️⃣ Word Cloud
# wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(all_words))
# plt.figure(figsize=(15,7))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.title("Word Cloud of IMDB Reviews")
# plt.show()

# # 9️⃣ Sentiment-specific analysis: Average review length per sentiment
# sentiment_stats = df.groupby('sentiment')[['review_word_count','review_char_count']].mean()
# print("\nAverage review length per sentiment:\n", sentiment_stats)