# IMDB Sentiment Analysis

This repository implements an sentiment analysis for movie reviews (IMDB). It demonstrates text cleaning, tokenization (Keras Tokenizer), bag-of-words (CountVectorizer), TF-IDF features, and classic classifiers (MultinomialNB and SVM) with training and evaluation.

---

## Dependencies

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
```

If using `nltk` stopwords:

```python
import nltk
nltk.download('stopwords')
```

---

## Dataset

You can use the IMDB dataset from Keras or download from Kaggle. 

---

## Preprocessing

Example text cleaning function:

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)         # remove URLs
    text = re.sub(r"<.*?>", "", text)                       # remove HTML tags
    text = re.sub(r"[^a-z\s]", "", text)                   # keep letters only
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

reviews['review'] = reviews['review'].apply(clean_text)
```

---

## Train / Test Split

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

---

## Training Models

### Multinomial Naive Bayes

Multinomial Naive Bayes Accuracy: **0.859836**

### Support Vector Machine (SVC)

 SVC Accuracy: **0.863365**

> Tip: For SVM training speed on sparse TF-IDF, use a `LinearSVC` (from `sklearn.svm import LinearSVC`) or sample a subset (e.g., 10k rows) before training.

---
