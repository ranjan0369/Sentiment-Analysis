import pandas as pd
import re
import nltk
nltk.data.path.append("/Users/ranjan/US/projects/Sentiment-Analysis/nltk_data")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv("imdb_dataset.csv")

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'<br\s*/?>', ' ', text)          # remove <br/> tags
    text = text.lower()                              # lowercase
    text = re.sub(r'[^a-z\s]', '', text)            # remove special chars
    tokens = text.split()                             # tokenize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words] # remove stopwords and lemmatize
    return " ".join(tokens)

df['clean_review'] = df['review'].apply(preprocess_text)
# Save preprocessed data
df.to_csv("imdb_dataset_cleaned.csv", index=False)
print("Preprocessing complete. Cleaned data saved to 'imdb_dataset_cleaned.csv'")