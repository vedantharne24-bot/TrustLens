import pandas as pd
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

nltk.download('stopwords')
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

print("1. Loading dataset...")
df = pd.read_csv("fake_reviews_dataset.csv")

# Ensure column names match your CSV
df = df[['text_', 'label']]
df.dropna(inplace=True)

# Map labels: CG (Fake) -> 1, OR (Real) -> 0
df['label'] = df['label'].map({'CG': 1, 'OR': 0})

# Check for two classes to prevent the ValueError seen in your screenshot
if len(df['label'].unique()) < 2:
    print("Error: Dataset must contain both CG (Fake) and OR (Real) labels.")
    exit()

def clean_text(text):
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower().split()
    return ' '.join([ps.stem(w) for w in review if w not in stop_words])

print("2. Cleaning data...")
df['cleaned'] = df['text_'].apply(clean_text)

# 3. Create Pipeline (Bundles Vectorizer + Model)
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1,2))),
    ('model', LogisticRegression(max_iter=2000))
])

X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['label'], test_size=0.2, stratify=df['label'])

print("4. Training Pipeline...")
pipeline.fit(X_train, y_train)

# Save the entire pipeline as model.pkl
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print(f"✅ Success! Accuracy: {accuracy_score(y_test, pipeline.predict(X_test))*100:.2f}%")