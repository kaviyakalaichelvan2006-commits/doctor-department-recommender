from flask import Flask, request, jsonify
from flask_cors import CORS
from model import predict_department

app = Flask(__name__)
CORS(app)

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    symptom = data.get("symptom")

    # Input validation
    if not symptom:
        return jsonify({
            "error": "Please enter symptom"
        })

    result = predict_department(symptom)

    return jsonify({
        "symptom": symptom,
        "department": result
    })

if __name__ == "__main__":
    app.run(debug=True)