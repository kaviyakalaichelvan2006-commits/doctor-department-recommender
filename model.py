import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("dataset.csv")

# Inputs and outputs
X = data["symptom"]
y = data["department"]

# Convert text to vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Prediction function
def predict_department(symptom):

    vec = vectorizer.transform([symptom])

    prediction = model.predict(vec)[0]

    return prediction