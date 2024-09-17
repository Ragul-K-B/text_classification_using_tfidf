import joblib
import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# Load spaCy model for word vectors
nlp = spacy.load("en_core_web_lg")

# Load the dataset
df = pd.read_csv("PashtoCorpusUpdated.csv")

X = df['text']
Y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=110)

# Initialize TF-IDF vectorizer
tfidf = TfidfVectorizer()

# Fit and transform the training data
X_train_vector = tfidf.fit_transform(X_train)

# Function to vectorize texts using spaCy and TF-IDF
def vectorising(texts, nlp, tfidf):
    text_vector_final = []
    text_vector = tfidf.transform(texts.tolist())
    text_feature = tfidf.get_feature_names_out()

    for i, text in enumerate(texts):
        word_spacy_vector = []
        words = text.split()

        for word in words:
            word_vector = nlp(word).vector
            if word.lower() in text_feature:
                word_index = list(text_feature).index(word.lower())
                word_vector_tfidf = text_vector[i, word_index]
                weighted_vector = word_vector * word_vector_tfidf
                word_spacy_vector.append(weighted_vector)
            else:
                word_spacy_vector.append(word_vector)

        if word_spacy_vector:
            text_vector_final.append(np.mean(word_spacy_vector, axis=0))
        else:
            text_vector_final.append(np.zeros(nlp.vocab.vectors_length))

    return np.array(text_vector_final)

# Vectorize training and testing sets using TF-IDF + spaCy
X_train_tfidf_spacy = vectorising(X_train, nlp, tfidf)
X_test_tfidf_spacy = vectorising(X_test, nlp, tfidf)

# Train the SVM model
model = SVC(kernel='linear')
model.fit(X_train_tfidf_spacy, y_train)

# Save the trained model and vectorizer
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/svm_model.pkl')
joblib.dump(tfidf, 'model/tfidf_vectorizer.pkl')

# Predict and evaluate on the test set
y_pred = model.predict(X_test_tfidf_spacy)
accuracy = accuracy_score(y_test, y_pred)
print(f'Model accuracy on test set: {accuracy * 100:.2f}%')
