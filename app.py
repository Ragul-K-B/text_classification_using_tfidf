from flask import Flask, render_template, request
import joblib
import re
import spacy
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")

# Load the saved model and vectorizer
model = joblib.load('model/svm_model.pkl')
tfidf = joblib.load('model/tfidf_vectorizer.pkl')

# Function to vectorize user input
def vectorising_for_user(text, nlp, tfidf):
    text_vector_final = []
    text_vector = tfidf.transform([text])
    text_feature = tfidf.get_feature_names_out()

    word_spacy_vector = []
    words = text.split()

    for word in words:
        word_vector = nlp(word).vector
        if word.lower() in text_feature:
            word_index = list(text_feature).index(word.lower())
            word_vector_tfidf = text_vector[0, word_index]
            weighted_vector = word_vector * word_vector_tfidf
            word_spacy_vector.append(weighted_vector)
        else:
            word_spacy_vector.append(word_vector)

    if word_spacy_vector:
        text_vector_final.append(np.mean(word_spacy_vector, axis=0))
    else:
        text_vector_final.append(np.zeros(nlp.vocab.vectors_length))

    return np.array(text_vector_final)

# Route for home page
@app.route('/')
def home():
    # Set initial image to sentiment.png (neutral/default)
    return render_template('index.html', image_name='sentiment.png')

# Route to process form submission
@app.route('/classify', methods=['POST'])
def classify():
    if request.method == 'POST':
        user_input = request.form['sentence']
        user_input = re.sub('[^\w\s]', '', user_input.lower())  # Preprocess input

        # Vectorize input for prediction
        user_input_vector = vectorising_for_user(user_input, nlp, tfidf)
        prediction = model.predict(user_input_vector)[0]

        # Interpret prediction and assign corresponding image
        if prediction == 1:
            sentiment = 'Positive'
            image_name = 'positive.png'  # Use the positive image
        elif prediction == -1:
            sentiment = 'Negative'
            image_name = 'negative.png'  # Use the negative image
        else:
            sentiment = 'Neutral'
            image_name = 'sentiment.png'  # Default image for neutral

        return render_template('index.html', sentence=user_input, sentiment=sentiment, image_name=image_name)

if __name__ == '__main__':
    app.run(debug=True)
