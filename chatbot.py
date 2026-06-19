import pandas as pd
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

faq_data = pd.read_csv("faq_data.csv")

questions = faq_data["question"]

def preprocess(text):
    text = text.lower()

    tokens = word_tokenize(text)

    tokens = [
        word for word in tokens
        if word not in stopwords.words("english")
        and word not in string.punctuation
    ]

    return " ".join(tokens)

processed_questions = questions.apply(preprocess)

vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(processed_questions)

def get_response(user_input):

    processed_input = preprocess(user_input)

    user_vector = vectorizer.transform([processed_input])

    similarity_scores = cosine_similarity(
        user_vector,
        faq_vectors
    )

    best_match = similarity_scores.argmax()

    score = similarity_scores[0][best_match]

    if score < 0.2:
        return "Sorry, I couldn't find a matching answer."

    return faq_data.iloc[best_match]["answer"]