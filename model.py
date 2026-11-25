import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("dataset.csv")


# Inputs and outputs
X = data["symptoms"] + " " + data["department"]

# Predict doctor directly
y = data["doctor_name"]


# Convert text into vectors
vectorizer = TfidfVectorizer()

X_vec = vectorizer.fit_transform(X)


# Train model
model = MultinomialNB()

model.fit(X_vec, y)


# Accuracy
predictions = model.predict(X_vec)

accuracy = accuracy_score(y, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Prediction function
def predict_department(symptom):

    # Lowercase input
    processed_input = symptom.lower()

    # Convert into vector
    vec = vectorizer.transform([processed_input])

    # Prediction probabilities
    probs = model.predict_proba(vec)[0]

    # Doctor names
    classes = model.classes_

    # Top 3 recommendations
    top_indices = probs.argsort()[-3:][::-1]

    results = []

    for i in top_indices:

        confidence = round(probs[i] * 300)

        if confidence >= 1:

            # Predicted doctor
            doctor_name = classes[i]

            # Get doctor details
            doctor_data = data[
                data["doctor_name"] == doctor_name
            ].iloc[0]

            results.append({

                "department": str(
                    doctor_data["department"]
                ),

                "doctor": str(
                    doctor_data["doctor_name"]
                ),

                "experience": int(
                    doctor_data["experience"]
                ),

                "availability": str(
                    doctor_data["availability"]
                ),

                "rating": float(
                    doctor_data["rating"]
                ),

                "confidence": int(confidence)
            })

    return results