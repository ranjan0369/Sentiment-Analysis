import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

df=pd.read_csv("dataset.csv")
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
# positive_text = " ".join(df[df['sentiment']=='positive']['review'])
# negative_text = " ".join(df[df['sentiment']=='negative']['review'])

df['review_word_count'] =df['review'].apply(lambda x: len(str(x).split()))

# plt.figure(figsize=(10,5))
# plt.subplot(1,2,1)
# sns.histplot(df['word_count'], bins=50, kde=True, color='green')
# plt.title('Word Count Distribution')
# plt.xlabel('Word Count')

# plt.subplot(1,2,2)
# sns.boxplot(x="sentiment", y="word_count", data=df)
# plt.title("Word Count by Sentiment")
# plt.xlabel("Sentiment")
# plt.ylabel("Word Count")

# plt.tight_layout()
# plt.show()

# # Positive word cloud
# wc = WordCloud(width=800, height=400, background_color="white").generate(positive_text)
# plt.figure(figsize=(10,5))
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.title("Positive Reviews Word Cloud")
# plt.show()

# # Negative word cloud
# wc = WordCloud(width=800, height=400, background_color="black", colormap="Reds").generate(negative_text)
# plt.figure(figsize=(10,5))
# plt.imshow(wc, interpolation="bilinear")
# plt.axis("off")
# plt.title("Negative Reviews Word Cloud")
# plt.show()

df['review_char_count'] = df['review'].apply(lambda x: len(str(x)))
corr = df[['review_word_count','review_char_count','sentiment']].corr()

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.show()
