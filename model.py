import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
data = pd.read_csv("dataset.csv")

# Inputs and outputs
X = data["symptom"]
y = data["department"]

# Convert text into numerical vectors
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = MultinomialNB()
model.fit(X_vec, y)

# Accuracy calculation
predictions = model.predict(X_vec)
accuracy = accuracy_score(y, predictions)

print("Model Accuracy:", round(accuracy * 100, 2), "%")


# Prediction function
def predict_department(symptom):

    # Convert input to lowercase
    processed_input = symptom.lower()

    # Convert user input into vector
    vec = vectorizer.transform([processed_input])

    # Get probabilities
    probs = model.predict_proba(vec)[0]

    # Department names
    classes = model.classes_

    # Get Top-3 predictions
    top_indices = probs.argsort()[-3:][::-1]

    results = []

    for i in top_indices:

        confidence = round(probs[i] * 100)

        # Filter weak predictions
        if confidence >= 10:

            department = classes[i]

            # Get doctor name from dataset
            doctor = data[data["department"] == department]["doctor"].values[0]

            results.append({
                "department": department,
                "doctor": doctor,
                "confidence": confidence
            })

    return results