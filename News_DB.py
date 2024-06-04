import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

df = pd.read_csv('bbc-news-data.csv', sep='\t')

df = df.dropna(subset=['content'])
df['content'] = df['content'].str.lower()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def process_text(text):
    words = word_tokenize(text)
    words = [word for word in words if word.isalpha()]  # Keep only words
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

df['cleaned_content'] = df['content'].apply(process_text)