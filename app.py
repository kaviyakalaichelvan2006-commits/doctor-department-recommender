from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_department

app = Flask(__name__)
CORS(app)


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    symptom = data.get("symptom")

    # Input validation
    if not symptom or symptom.strip() == "":
        return jsonify({
            "error": "Please enter a valid symptom"
        }), 400

    results = predict_department(symptom)

    # No strong match
    if results[0]["confidence"] < 10:
        return jsonify({
            "message": "No strong recommendation found"
        })

    return jsonify({
        "input": symptom,
        "recommendations": results,
        "note": "Prediction based on symptom similarity using ML model",
        "feedback": "Was this recommendation helpful?"
    })


# Feedback API
@app.route("/feedback", methods=["POST"])
def feedback():

    data = request.get_json()

    user_feedback = data.get("feedback")

    # Store feedback
    with open("feedback.txt", "a") as f:
        f.write(user_feedback + "\n")

    return jsonify({
        "message": "Feedback received successfully"
    })


if __name__ == "__main__":
    app.run(debug=True)