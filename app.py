from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("ai4lassa_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse JSON input
        data = request.get_json()
        features = [
            data["fever"],
            data["bleeding"],
            data["headache"],
            data["vomiting"],
            data["temperature"]
        ]
        
        # Scale input and predict
        features_scaled = scaler.transform([features])
        prediction = model.predict_proba(features_scaled)[0]
        class_1 = prediction[1]

        return jsonify({"prediction": class_1})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)