from flask import Flask, request, jsonify, g
import joblib
import numpy as np
import psycopg2 # type: ignore
import os
import bcrypt # type: ignore
import datetime
from functools import wraps
from jose import jwt #type: ignore

# Load model and scaler
model = joblib.load("ai4lassa_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# Environment variables
DATABASE_URL = os.environ.get("DATABASE_URL", "your_local_fallback")
SECRET_KEY = os.environ.get("SECRET_KEY", "change_this_key")
JWT_ALGORITHM = "HS256"

# DB connection
conn = psycopg2.connect(DATABASE_URL, sslmode='require')
cursor = conn.cursor()

# ========== Helper Functions ==========
def generate_token(user_id, role, state):
    payload = {
        "user_id": user_id,
        "role": role,
        "state": state,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token):
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])

def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = request.headers.get("Authorization")
        if not token:
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = token.replace("Bearer ", "")
            decoded = decode_token(token)
            g.user = decoded
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401
        return f(*args, **kwargs)
    return decorated

# ========== Endpoints ==========

# 🧪 Prediction
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [data["fever"], data["bleeding"], data["headache"], data["vomiting"], data["temperature"]]
        features_scaled = scaler.transform([features])
        prediction = model.predict_proba(features_scaled)[0]
        return jsonify({"prediction": prediction[1]})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 📥 Upload stats (auth protected)
@app.route("/upload_stats", methods=["POST"])
@token_required
def upload_stats():
    try:
        user_role = g.user["role"]
        user_state = g.user["state"]

        data = request.get_json()
        state = data["state"]
        if user_role != "superadmin" and user_state != state:
            return jsonify({"error": "Unauthorized for this state"}), 403

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
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 👀 View history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        state = request.args.get("state")
        start_year = int(request.args.get("start_year"))
        end_year = int(request.args.get("end_year"))
        start_month = int(request.args.get("start_month"))
        end_month = int(request.args.get("end_month"))

        cursor.execute("""
            SELECT year, month, cases, deaths, recoveries
            FROM lassa_stats
            WHERE state = %s
                       AND year BETWEEN %s AND %s
                       AND month BETWEEN %s AND %s
            ORDER BY year, month
        """, (state, start_year, end_year, start_month, end_month))
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

# 🛠️ Create table
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
            );
            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role VARCHAR(50) NOT NULL,
                state VARCHAR(50)
            );
        """)
        conn.commit()
        return "Tables created successfully"
    except Exception as e:
        conn.rollback()
        return f"Error: {str(e)}", 500

# 👤 Register user
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]
        role = data["role"]
        state = data.get("state")

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        cursor.execute("""
            INSERT INTO users (email, password, role, state)
            VALUES (%s, %s, %s, %s)
        """, (email, hashed_pw, role, state))
        conn.commit()
        return jsonify({"message": "User registered successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 🔐 Login
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]

        cursor.execute("SELECT id, password, role, state FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404

        user_id, hashed_pw, role, state = user
        if not bcrypt.checkpw(password.encode(), hashed_pw.encode()):
            return jsonify({"error": "Incorrect password"}), 401

        token = generate_token(user_id, role, state)
        return jsonify({"token": token})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🧪 Test auth
@app.route("/me", methods=["GET"])
@token_required
def me():
    return jsonify({"user": g.user})

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
