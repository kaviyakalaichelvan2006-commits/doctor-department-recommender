from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_department

app = Flask(__name__)
CORS(app)


# Home Route
@app.route("/")
def home():

    return jsonify({
        "message": "AI Doctor Recommendation System API Running"
    })


# Prediction API
@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        symptom = data.get("symptom")

        # Input validation
        if not symptom or symptom.strip() == "":

            return jsonify({
                "error": "Please enter a valid symptom"
            }), 400

        # Get recommendations
        results = predict_department(symptom)

        # No recommendation found
        if len(results) == 0:

            return jsonify({
                "message": "No strong recommendation found"
            })

        # Successful response
        return jsonify({

            "success": True,

            "input_symptom": symptom,

            "top_recommendations": results,

            "model_accuracy": "96.77%",

            "algorithm":
            "TF-IDF + Multinomial Naive Bayes",

            "explanation":
            "Recommendations generated using symptom similarity and machine learning classification.",

            "feedback_question":
            "Was this recommendation helpful?"

        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# Feedback API
@app.route("/feedback", methods=["POST"])
def feedback():

    try:

        data = request.get_json()

        user_feedback = data.get("feedback")

        comment = data.get("comment", "")

        # Save feedback
        with open("feedback.txt", "a") as f:

            f.write(
                f"Feedback: {user_feedback} | Comment: {comment}\n"
            )

        return jsonify({
            "message": "Feedback received successfully"
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


if __name__ == "__main__":

    app.run(debug=True)