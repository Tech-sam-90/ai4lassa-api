from flask import Flask, request, jsonify, g
import joblib
import numpy as np
import psycopg2  # type: ignore
import os
import bcrypt  # type: ignore
import datetime
from functools import wraps
from jose import jwt  # type: ignore

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
def generate_token(user_id, role, state, is_superadmin, is_active):
    payload = {
        "user_id": user_id,
        "role": role,
        "state": state,
        "is_superadmin": is_superadmin,
        "is_active": is_active,
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
        class_1 = prediction[1]
        percentage = round(class_1 * 100, 2)
        return jsonify({"prediction": percentage})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 📥 Upload stats (auth protected)
@app.route("/upload_stats", methods=["POST"])
@token_required
def upload_stats():
    try:
        user_role = g.user["role"]
        user_state = g.user["state"]
        is_active = g.user["is_active"]

        if not is_active:
            return jsonify({"error": "Account not activated by superadmin"}), 403

        data = request.get_json()
        if isinstance(data, dict):
            data = [data]

        for entry in data:
            state = entry["state"]
            if user_role != "superadmin" and user_state != state:
                return jsonify({"error": f"Unauthorized for state {state}"}), 403

            year = entry["year"]
            month = entry["month"]
            cases = entry["cases"]
            deaths = entry["deaths"]
            recoveries = entry["recoveries"]

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
        result = [
            {
                "year": row[0],
                "month": row[1],
                "cases": row[2],
                "deaths": row[3],
                "recoveries": row[4]
            }
            for row in rows
        ]
        return jsonify(result)
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 🛠️ Create table
@app.route("/create_table", methods=["GET"])
def create_table():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()

        cursor.execute("SELECT current_database(), current_schema();")
        db_info = cursor.fetchone()
        print(f"Connected to DB: {db_info[0]}, Schema: {db_info[1]}")
        cursor.execute("""
            DROP TABLE IF EXISTS lassa_stats CASCADE;
            DROP TABLE IF EXISTS users CASCADE;
                                  
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
                state VARCHAR(50),
                is_superadmin BOOLEAN DEFAULT FALSE,
                is_active BOOLEAN DEFAULT FALSE
            );
        """)
        conn.commit()
        return "Tables dropped and recreated successfully"
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
        is_superadmin = data.get("is_superadmin", False)

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        is_active = True if is_superadmin else False

        cursor.execute("""
            INSERT INTO users (email, password, role, state, is_superadmin, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (email, hashed_pw, role, state, is_superadmin, is_active))
        conn.commit()

        if is_superadmin:
            return jsonify({"message":"Superadmin account created and activated successfully"})
        else:
            return jsonify({"message": "User registered successfully. Awaiting superadmin activation."})
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

        cursor.execute("SELECT id, password, role, state, is_superadmin, is_active FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "User not found"}), 404

        user_id, hashed_pw, role, state, is_superadmin, is_active = user
        if not bcrypt.checkpw(password.encode(), hashed_pw.encode()):
            return jsonify({"error": "Incorrect password"}), 401

        token = generate_token(user_id, role, state, is_superadmin, is_active)
        return jsonify({"token": token})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🧪 Test auth
@app.route("/me", methods=["GET"])
@token_required
def me():
    return jsonify({"user": g.user})

# 🔓 Activate user (Superadmin only)
@app.route("/activate_user/<int:user_id>", methods=["POST"])
@token_required
def activate_user(user_id):
    try:
        if not g.user.get("is_superadmin", False):
            return jsonify({"error": "Only superadmins can activate accounts"}), 403

        cursor.execute("UPDATE users SET is_active = TRUE WHERE id = %s", (user_id,))
        conn.commit()
        return jsonify({"message": "User activated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
