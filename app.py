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

def superadmin_required(f):
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if not g.user.get("is_superadmin", False):
            return jsonify({"error": "Superadmin access required"}), 403
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

        updated_records = []
        inserted_records = []        

        for entry in data:
            state = entry["state"]
            if user_role != "superadmin" and user_state != state:
                return jsonify({"error": f"Unauthorized for state {state}"}), 403

            year = entry["year"]
            month = entry["month"]
            cases = entry["cases"]
            deaths = entry["deaths"]
            recoveries = entry["recoveries"]
            
            #Check if recors already exists
            cursor.execute("""
                SELECT id FROM lassa_stats
                WHERE state = %s AND year = %s AND month = %s
            """, (state, year, month))
            existing_record = cursor.fetchone()

            if existing_record:
                #Update existing record
                cursor.execute("""
                    UPDATE lassa_stats
                    SET cases = %s, deaths = %s, recoveries = %s
                    WHERE state = %s AND year = %s AND month = %s
                """, (cases, deaths, recoveries, state, year, month))
                updated_records.append(f"{state} {year}-{month:02d}")
            else:
                #Insert new records
                cursor.execute("""
                    INSERT INTO lassa_stats (state, year, month, cases, deaths, recoveries)
                    VALUES (%s, %s, %s, %s, %s, %s)
                """, (state, year, month, cases, deaths, recoveries))
                inserted_records.append(f"{state} {year}-{month:02d}")


        conn.commit()

        #Prepare response message
        message_parts = []
        if inserted_records:
            message_parts.append(f"Inserted {len(inserted_records)} new records")
        if updated_records:
            message_parts.append(f"Updated {len(updated_records)} existing records")
        
        return jsonify({
            "message": ". ".join(message_parts),
            "inserted": inserted_records,
            "updated": updated_records,
            "total_processed": len(data)
        })

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
                is_active BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        return "Tables dropped and recreated successfully"
    except Exception as e:
        conn.rollback()
        return f"Error: {str(e)}", 500

# 👤 Register state admin (app registration - requires approval)
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]
        role = data.get("role", "state_admin")  # Default to state_admin
        state = data.get("state")

        # Prevent superadmin registration through app
        if role == "superadmin" or data.get("is_superadmin", False):
            return jsonify({"error": "Superadmin accounts cannot be created through this endpoint"}), 403

        # Validate required fields for state_admin
        if role == "state_admin" and not state:
            return jsonify({"error": "State is required for state admin registration"}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        cursor.execute("""
            INSERT INTO users (email, password, role, state, is_superadmin, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (email, hashed_pw, role, state, False, False))
        conn.commit()

        return jsonify({"message": "State admin registered successfully. Awaiting superadmin activation."})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 👑 Manual superadmin registration (Postman only)
@app.route("/create_superadmin", methods=["POST"])
def create_superadmin():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]
        
        # Optional secret key validation for extra security
        secret_key = data.get("secret_key")
        expected_secret = os.environ.get("SUPERADMIN_SECRET_KEY", "your_super_secret_key")
        
        if secret_key != expected_secret:
            return jsonify({"error": "Invalid secret key"}), 403

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        cursor.execute("""
            INSERT INTO users (email, password, role, state, is_superadmin, is_active)
            VALUES (%s, %s, %s, %s, %s, %s)
        """, (email, hashed_pw, "superadmin", None, True, True))
        conn.commit()

        return jsonify({"message": "Superadmin account created and activated successfully"})
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
        return jsonify({"token": token, "user": {"role": role, "state": state, "is_active": is_active}})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🧪 Test auth
@app.route("/me", methods=["GET"])
@token_required
def me():
    return jsonify({"user": g.user})

# 📋 Get pending users (Superadmin only)
@app.route("/pending_users", methods=["GET"])
@superadmin_required
def get_pending_users():
    try:
        cursor.execute("""
            SELECT id, email, role, state, created_at 
            FROM users 
            WHERE is_active = FALSE AND is_superadmin = FALSE
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        
        result = [
            {
                "id": row[0],
                "email": row[1],
                "role": row[2],
                "state": row[3],
                "created_at": row[4].isoformat() if row[4] else None
            }
            for row in rows
        ]
        return jsonify({"pending_users": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔓 Activate user (Superadmin only)
@app.route("/activate_user/<int:user_id>", methods=["POST"])
@superadmin_required
def activate_user(user_id):
    try:
        cursor.execute("UPDATE users SET is_active = TRUE WHERE id = %s AND is_superadmin = FALSE", (user_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"error": "User not found or user is already a superadmin"}), 404
            
        conn.commit()
        return jsonify({"message": "User activated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 🚫 Deactivate user (Superadmin only)
@app.route("/deactivate_user/<int:user_id>", methods=["POST"])
@superadmin_required
def deactivate_user(user_id):
    try:
        cursor.execute("UPDATE users SET is_active = FALSE WHERE id = %s AND is_superadmin = FALSE", (user_id,))
        
        if cursor.rowcount == 0:
            return jsonify({"error": "User not found or user is already a superadmin"}), 404
            
        conn.commit()
        return jsonify({"message": "User deactivated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 👥 Get all users (Superadmin only)
@app.route("/users", methods=["GET"])
@superadmin_required
def get_all_users():
    try:
        cursor.execute("""
            SELECT id, email, role, state, is_active, created_at 
            FROM users 
            WHERE is_superadmin = FALSE
            ORDER BY created_at DESC
        """)
        rows = cursor.fetchall()
        
        result = [
            {
                "id": row[0],
                "email": row[1],
                "role": row[2],
                "state": row[3],
                "is_active": row[4],
                "created_at": row[5].isoformat() if row[5] else None
            }
            for row in rows
        ]
        return jsonify({"users": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 👨‍💼 Get all admins (Superadmin only)
@app.route("/admins", methods=["GET"])
@superadmin_required
def get_all_admins():
    try:
        cursor.execute("""
            SELECT id, email, role, state, is_superadmin, is_active, created_at 
            FROM users 
            ORDER BY is_superadmin DESC, created_at DESC
        """)
        rows = cursor.fetchall()
        
        result = [
            {
                "id": row[0],
                "email": row[1],
                "role": row[2],
                "state": row[3],
                "is_superadmin": row[4],
                "is_active": row[5],
                "created_at": row[6].isoformat() if row[6] else None
            }
            for row in rows
        ]
        return jsonify({"admins": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔓 Activate admin (Superadmin only)
@app.route("/admins/<int:admin_id>/activate", methods=["PATCH"])
@superadmin_required
def activate_admin(admin_id):
    try:
        # Check if the target user exists and is not the current superadmin
        cursor.execute("SELECT id, email, is_superadmin FROM users WHERE id = %s", (admin_id,))
        target_user = cursor.fetchone()
        
        if not target_user:
            return jsonify({"error": "Admin not found"}), 404
            
        # Prevent superadmin from deactivating themselves (safety check)
        if target_user[2] and target_user[0] == g.user["user_id"]:
            return jsonify({"error": "Cannot modify your own superadmin account"}), 403
        
        cursor.execute("UPDATE users SET is_active = TRUE WHERE id = %s", (admin_id,))
        conn.commit()
        
        return jsonify({"message": f"Admin {target_user[1]} activated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 🚫 Deactivate admin (Superadmin only)
@app.route("/admins/<int:admin_id>/deactivate", methods=["PATCH"])
@superadmin_required
def deactivate_admin(admin_id):
    try:
        # Check if the target user exists and is not the current superadmin
        cursor.execute("SELECT id, email, is_superadmin FROM users WHERE id = %s", (admin_id,))
        target_user = cursor.fetchone()
        
        if not target_user:
            return jsonify({"error": "Admin not found"}), 404
            
        # Prevent superadmin from deactivating themselves
        if target_user[2] and target_user[0] == g.user["user_id"]:
            return jsonify({"error": "Cannot deactivate your own superadmin account"}), 403
            
        cursor.execute("UPDATE users SET is_active = FALSE WHERE id = %s", (admin_id,))
        conn.commit()
        
        return jsonify({"message": f"Admin {target_user[1]} deactivated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 👑 Promote admin to superadmin (Superadmin only)
@app.route("/admins/<int:admin_id>/make_super", methods=["PATCH"])
@superadmin_required
def promote_to_superadmin(admin_id):
    try:
        # Check if the target user exists and is not already a superadmin
        cursor.execute("SELECT id, email, is_superadmin, is_active FROM users WHERE id = %s", (admin_id,))
        target_user = cursor.fetchone()
        
        if not target_user:
            return jsonify({"error": "Admin not found"}), 404
            
        if target_user[2]:  # is_superadmin is True
            return jsonify({"error": "User is already a superadmin"}), 400
            
        # Promote to superadmin and ensure they're active
        cursor.execute("""
            UPDATE users 
            SET is_superadmin = TRUE, is_active = TRUE, role = 'superadmin', state = NULL 
            WHERE id = %s
        """, (admin_id,))
        conn.commit()
        
        return jsonify({"message": f"Admin {target_user[1]} promoted to superadmin successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400
    

# add an endpoint to for calculating the overall total for each state
@app.route("/state_totals", methods=["GET"])
def state_totals():
    try:
        cursor.execute("""
            SELECT state, SUM(cases) AS total_cases
            FROM lassa_stats
            GROUP BY state
            ORDER BY total_cases DESC
        """)
        rows = cursor.fetchall()

        result = [
            {"state": row[0], "total_cases": row[1]}
            for row in rows
        ]

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)