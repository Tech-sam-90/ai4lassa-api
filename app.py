
from flask import Flask, request, jsonify
import joblib
import numpy as np
import psycopg2
import os

# Load model and scaler
model = joblib.load("ai4lassa_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Setup PostgreSQL connection
DATABASE_URL = os.environ.get("DATABASE_URL", "your_local_fallback_if_needed")
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()

# ML Prediction Endpoint
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [
            data["fever"],
            data["bleeding"],
            data["headache"],
            data["vomiting"],
            data["temperature"]
        ]
        features_scaled = scaler.transform([features])
        prediction = model.predict_proba(features_scaled)[0]
        class_1 = prediction[1]
        return jsonify({"prediction": class_1})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Admin Upload Stats Endpoint
@app.route("/upload_stats", methods=["POST"])
def upload_stats():
    try:
        data = request.get_json()
        state = data["state"]
        year = data["year"]
        month = data["month"]
        cases = data["cases"]
        deaths = data["deaths"]
        recoveries = data["recoveries"]

        cursor.execute("""
            INSERT INTO lassa_stats (state, year, month, cases, deaths, recoveries)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (state, year, month, cases, deaths, recoveries))
        conn.commit()
        return jsonify({"message": "Data uploaded successfully"})
    except Exception as e:
        conn.rollback()  # <- Prevents "transaction aborted" issues
        return jsonify({"error": str(e)}), 400

# User View History Endpoint
@app.route("/history", methods=["GET"])
def get_history():
    try:
        state = request.args.get("state")
        start_year = int(request.args.get("start_year"))
        end_year = int(request.args.get("end_year"))

        cursor.execute("""
            SELECT year, month, cases, deaths, recoveries
            FROM lassa_stats
            WHERE state = %s AND year BETWEEN %s AND %s
            ORDER BY year, month
        """, (state, start_year, end_year))

        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append({
                "year": row[0],
                "month": row[1],
                "cases": row[2],
                "deaths": row[3],
                "recoveries": row[4]
            })
        return jsonify(result)
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# Optional: Create the stats table once
@app.route("/create_table")
def create_table():
    try:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS lassa_stats (
                id SERIAL PRIMARY KEY,
                state VARCHAR(50),
                year INT,
                month INT,
                cases INT,
                deaths INT,
                recoveries INT
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role VARCHAR(50) NOT NULL,
                state VARCHAR(50) DEFAULT NULL
            )
        """)
        conn.commit()
        return "Table created successfully"
    except Exception as e:
        conn.rollback()
        return f"Failed to create table: {str(e)}", 500

# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
