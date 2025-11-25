from flask import Flask, request, jsonify, g
import joblib
import numpy as np
import psycopg2  # type: ignore
import os
import bcrypt  # type: ignore
import datetime
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps
from jose import jwt  # type: ignore
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO)

# Load model and scaler
model = joblib.load("ai4lassa_svm_model.pkl")
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)

# ========== Configuration ==========

# Environment variables
DATABASE_URL = os.environ.get("DATABASE_URL")
SECRET_KEY = os.environ.get("SECRET_KEY", "change_this_key")
SUPERADMIN_SECRET_KEY = os.environ.get("SUPERADMIN_SECRET_KEY", "your_super_secret_key")
JWT_ALGORITHM = "HS256"

# Email configuration from environment variables
EMAIL_HOST = os.environ.get("EMAIL_HOST")  # e.g., 'smtp.gmail.com'
EMAIL_PORT = int(os.environ.get("EMAIL_PORT", 587))
EMAIL_USE_TLS = os.environ.get("EMAIL_USE_TLS", 'true').lower() in ['true', '1', 't']
EMAIL_USER = os.environ.get("EMAIL_USER")
EMAIL_PASS = os.environ.get("EMAIL_PASS")
SENDER_EMAIL = EMAIL_USER

# DB connection
try:
    conn = psycopg2.connect(DATABASE_URL, sslmode='verify-full')
    cursor = conn.cursor()
    logging.info("Database connection successful.")
except Exception as e:
    logging.error(f"Database connection failed: {e}")
    conn, cursor = None, None

# ========== Helper Functions ==========

def send_email(to_email, subject, html_content):
    """Sends an email using configured SMTP settings."""
    if not all([EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS]):
        logging.error("Email configuration is incomplete. Cannot send email.")
        # In a real app, you might want to raise an exception or handle this more gracefully
        return False
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SENDER_EMAIL
        msg['To'] = to_email
        msg['Subject'] = subject

        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
            if EMAIL_USE_TLS:
                server.starttls()
            server.login(EMAIL_USER, EMAIL_PASS)
            server.send_message(msg)
        logging.info(f"Email sent successfully to {to_email}.")
        return True
    except Exception as e:
        logging.error(f"Failed to send email to {to_email}: {e}")
        return False

def generate_token_and_hash():
    """Generates a secure token and its bcrypt hash."""
    token = secrets.token_urlsafe(32)
    token_hash = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
    return token, token_hash

def generate_numeric_token_and_hash(length=6):
    """Generates a secure numeric token and its bcrypt hash."""
    token = ''.join(secrets.choice('0123456789') for _ in range(length))
    token_hash = bcrypt.hashpw(token.encode(), bcrypt.gensalt()).decode()
    return token, token_hash

def generate_jwt(user_id, role, state, is_superadmin, is_active, is_verified):
    """Generates a JWT for authenticated sessions."""
    payload = {
        "user_id": user_id,
        "role": role,
        "state": state,
        "is_superadmin": is_superadmin,
        "is_active": is_active,
        "is_verified": is_verified,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=12)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=JWT_ALGORITHM)

def decode_token(token):
    """Decodes a JWT."""
    return jwt.decode(token, SECRET_KEY, algorithms=[JWT_ALGORITHM])

def token_required(f):
    """Decorator to protect routes with JWT authentication."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return jsonify({"error": "Token is missing"}), 401
        try:
            token = auth_header.replace("Bearer ", "")
            g.user = decode_token(token)
        except Exception as e:
            return jsonify({"error": f"Invalid token: {str(e)}"}), 401
        return f(*args, **kwargs)
    return decorated

def superadmin_required(f):
    """Decorator to restrict routes to superadmins."""
    @wraps(f)
    @token_required
    def decorated(*args, **kwargs):
        if not g.user.get("is_superadmin", False):
            return jsonify({"error": "Superadmin access required"}), 403
        return f(*args, **kwargs)
    return decorated

def log_upload_activity(user_id, action_details):
    """Logs upload activity for a user."""
    try:
        cursor.execute("""
            INSERT INTO upload_logs (user_id, action_details, created_at)
            VALUES (%s, %s, %s  )
        """, (user_id, action_details, datetime.datetime.now(datetime.timezone.utc)))
        conn.commit()
    except Exception as e:
        logging.error(f"Failed to log upload activity for user {user_id}: {e}")

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
    # Check if user is verified and activated by superadmin
    if not g.user.get("is_verified"):
        return jsonify({"error": "Account not verified. Please check your email."}), 403
    if not g.user.get("is_active"):
        return jsonify({"error": "Account not activated by superadmin."}), 403

    try:
        user_role = g.user["role"]
        user_state = g.user["state"]
        user_id = g.user["user_id"]
        data = request.get_json()
        if isinstance(data, dict):
            data = [data]

        updated_records, inserted_records = [], []
        for entry in data:
            state = entry["state"]
            if user_role != "superadmin" and user_state != state:
                return jsonify({"error": f"Unauthorized for state {state}"}), 403

            year, month, cases, deaths, recoveries = entry["year"], entry["month"], entry["cases"], entry["deaths"], entry["recoveries"]
            
            cursor.execute("SELECT id FROM lassa_stats WHERE state = %s AND year = %s AND month = %s", (state, year, month))
            if cursor.fetchone():
                cursor.execute("UPDATE lassa_stats SET cases = %s, deaths = %s, recoveries = %s WHERE state = %s AND year = %s AND month = %s",
                               (cases, deaths, recoveries, state, year, month))
                updated_records.append(f"{state} {year}-{month:02d}")
            else:
                cursor.execute("INSERT INTO lassa_stats (state, year, month, cases, deaths, recoveries) VALUES (%s, %s, %s, %s, %s, %s)",
                               (state, year, month, cases, deaths, recoveries))
                inserted_records.append(f"{state} {year}-{month:02d}")
        
        conn.commit()

        #Log the upload activity
        action_details = f"Inserted {len(inserted_records)} records, Updated {len(updated_records)} records. States: {','.join(set([entry['state'] for entry in data]))}"
        log_upload_activity(user_id, action_details)

        return jsonify({
            "message": f"Inserted {len(inserted_records)} and updated {len(updated_records)} records.",
            "inserted": inserted_records,
            "updated": updated_records
        })
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400
    
#Get last 5 uploads for the authenticated user
@app.route("/upload_stats/last_5", methods=["GET"])
@token_required
def get_my_recent_uploads():
    try:
        user_id = g.user["user_id"]
        cursor.execute("""
            SELECT action_details, created_at FROM upload_logs
            WHERE user_id = %s
            ORDER BY created_at DESC
            LIMIT 5
        """, (user_id,))

        uploads = []
        for row in cursor.fetchall():
            action_details, created_at = row
            uploads.append({
                "action_details": row[0],
                "created_at": row[1].isoformat()
            })
        return jsonify({"recent_uploads": uploads})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 👀 View history
@app.route("/history", methods=["GET"])
def get_history():
    try:
        args = request.args
        state, start_year, end_year, start_month, end_month = args.get("state"), int(args.get("start_year")), int(args.get("end_year")), int(args.get("start_month")), int(args.get("end_month"))
        
        cursor.execute("""
            SELECT year, month, cases, deaths, recoveries FROM lassa_stats
            WHERE state = %s AND year BETWEEN %s AND %s AND month BETWEEN %s AND %s
            ORDER BY year, month
        """, (state, start_year, end_year, start_month, end_month))
        
        result = [{"year": r[0], "month": r[1], "cases": r[2], "deaths": r[3], "recoveries": r[4]} for r in cursor.fetchall()]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🛠️ Create/Recreate Tables (for development)
@app.route("/create_tables", methods=["GET"])
def create_tables():
    try:
        cursor.execute("""
            DROP TABLE IF EXISTS upload_logs CASCADE;
            DROP TABLE IF EXISTS tokens CASCADE;
            DROP TABLE IF EXISTS lassa_stats CASCADE;
            DROP TABLE IF EXISTS users CASCADE;

            CREATE TABLE IF NOT EXISTS users (
                id SERIAL PRIMARY KEY,
                email VARCHAR(255) UNIQUE NOT NULL,
                password TEXT NOT NULL,
                first_name VARCHAR(100) NOT NULL,
                last_name VARCHAR(100) NOT NULL,
                role VARCHAR(50) NOT NULL,
                state VARCHAR(50),
                is_superadmin BOOLEAN DEFAULT FALSE,
                is_verified BOOLEAN DEFAULT FALSE, -- For email verification
                is_active BOOLEAN DEFAULT FALSE,   -- For superadmin activation
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS lassa_stats (
                id SERIAL PRIMARY KEY,
                state VARCHAR(50),
                year INT,
                month INT,
                cases INT,
                deaths INT,
                recoveries INT
            );

            CREATE TABLE IF NOT EXISTS tokens (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL,
                token_type VARCHAR(50) NOT NULL, -- 'verification' or 'password_reset'
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP WITH TIME ZONE NOT NULL
            );

            CREATE TABLE IF NOT EXISTS upload_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                action_details TEXT NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tokens_user_id ON tokens(user_id);
            CREATE INDEX IF NOT EXISTS idx_upload_logs_user_id ON upload_logs(user_id);
            CREATE INDEX IF NOT EXISTS idx_upload_logs_created_at ON upload_logs(created_at);
        """)
        conn.commit()
        return "Tables dropped and recreated successfully."
    except Exception as e:
        conn.rollback()
        return f"Error creating tables: {str(e)}", 500

# 👤 Register state admin
@app.route("/register", methods=["POST"])
def register():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]
        first_name = data["first_name"]
        last_name = data["last_name"]
        state = data.get("state")

        if not state:
            return jsonify({"error": "State is required for state admin registration"}), 400
        
        if not all([email, password, first_name, last_name]):
            return jsonify({"error": "All fields are required."}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

        # Insert user as unverified and inactive
        cursor.execute("""
            INSERT INTO users (email, password, first_name, last_name, role, state, is_superadmin, is_verified, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (email, hashed_pw, first_name, last_name, 'state_admin', state, False, False, False))
        user_id = cursor.fetchone()[0]

        # Generate and store verification token
        token, token_hash = generate_token_and_hash()
        expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=24)
        cursor.execute("""
            INSERT INTO tokens (user_id, token_hash, token_type, expires_at)
            VALUES (%s, %s, 'verification', %s)
        """, (user_id, token_hash, expires_at))
        
        # Send verification email
        verification_url = f"{request.host_url}verify_email?token={token}"
        html_content = f"""
            <h3>Welcome to AI4Lassa, {first_name} {last_name}!</h3>
            <p>Thank you for registering. Please click the link below to verify your email address:</p>
            <a href="{verification_url}">Verify My Email</a>
            <p>This link will expire in 24 hours.</p>
        """
        send_email(email, "Verify Your AI4Lassa Account", html_content)

        conn.commit()
        return jsonify({"message": "Registration successful. Please check your email to verify your account."})
    except psycopg2.IntegrityError:
        conn.rollback()
        return jsonify({"error": "An account with this email already exists."}), 409
    except Exception as e:
        conn.rollback()
        logging.error(f"Registration failed: {e}")
        return jsonify({"error": str(e)}), 400

# ✅ Verify Email
@app.route("/verify_email", methods=["GET"])
def verify_email():
    token = request.args.get('token')
    if not token:
        return jsonify({"error": "Verification token is missing."}), 400
    
    try:
        # Find a matching token hash (this is inefficient, better to find user first)
        # A better approach: JWT with user_id, but for now this works.
        cursor.execute("""
            SELECT id, user_id, token_hash, expires_at FROM tokens 
            WHERE token_type = 'verification' AND expires_at > NOW()
        """)
        all_tokens = cursor.fetchall()
        
        found_token = None
        for t in all_tokens:
            if bcrypt.checkpw(token.encode(), t[2].encode()):
                found_token = t
                break
        
        if not found_token:
            return jsonify({"error": "Invalid or expired verification token."}), 400
        
        token_id, user_id = found_token[0], found_token[1]
        
        # Activate user and delete token
        cursor.execute("UPDATE users SET is_verified = TRUE WHERE id = %s", (user_id,))
        cursor.execute("DELETE FROM tokens WHERE id = %s", (token_id,))
        
        conn.commit()
        return jsonify({"message": "Email verified successfully. Your account is now awaiting activation by a superadmin."})
    except Exception as e:
        conn.rollback()
        logging.error(f"Email verification failed: {e}")
        return jsonify({"error": "An error occurred during verification."}), 500

# 🔐 Login
@app.route("/login", methods=["POST"])
def login():
    try:
        data = request.get_json()
        email, password = data["email"], data["password"]

        cursor.execute("SELECT id, password, role, state, is_superadmin, is_verified, is_active FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "Invalid credentials"}), 401

        user_id, hashed_pw, role, state, is_superadmin, is_verified, is_active = user
        if not bcrypt.checkpw(password.encode(), hashed_pw.encode()):
            return jsonify({"error": "Invalid credentials"}), 401

        if not is_verified:
            return jsonify({"error": "Account not verified. Please check your email."}), 403
        
        # Superadmins don't need activation, but state_admins do
        if role == 'state_admin' and not is_active:
             return jsonify({"error": "Account has not been activated by a superadmin yet."}), 403

        token = generate_jwt(user_id, role, state, is_superadmin, is_active, is_verified)
        return jsonify({"token": token, "user": {"role": role, "state": state, "is_active": is_active, "is_superadmin": is_superadmin}})
    except Exception as e:
        logging.error(f"Login failed: {e}")
        return jsonify({"error": str(e)}), 400

# 🔑 Request Password Reset
@app.route("/request_password_reset", methods=["POST"])
def request_password_reset():
    try:
        email = request.json.get('email')
        if not email:
            return jsonify({"error": "Email is required"}), 400

        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            user_id = user[0]
            # Generate 6-digit code for password reset
            reset_code, token_hash = generate_numeric_token_and_hash()
            expires_at = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=15) # Short expiry

            # Store the hash of the reset code
            cursor.execute("DELETE FROM tokens WHERE user_id = %s AND token_type = 'password_reset'", (user_id,))
            cursor.execute("""
                INSERT INTO tokens (user_id, token_hash, token_type, expires_at)
                VALUES (%s, %s, 'password_reset', %s)
            """, (user_id, token_hash, expires_at))
            
            html_content = f"""
                <h3>Password Reset Request</h3>
                <p>You requested a password reset. Use the following code to reset your password:</p>
                <h2>{reset_code}</h2>
                <p>This code will expire in 15 minutes.</p>
            """
            send_email(email, "Your Password Reset Code", html_content)
            conn.commit()

        # Always return a generic message to prevent user enumeration
        return jsonify({"message": "If an account with that email exists, a password reset code has been sent."})
    except Exception as e:
        conn.rollback()
        logging.error(f"Password reset request failed: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# 🔄 Reset Password
@app.route("/reset_password", methods=["POST"])
def reset_password():
    try:
        data = request.json
        email, code, new_password = data.get('email'), data.get('code'), data.get('new_password')

        if not all([email, code, new_password]):
            return jsonify({"error": "Email, code, and new password are required."}), 400

        cursor.execute("SELECT id FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        if not user:
            return jsonify({"error": "Invalid code or email."}), 400
        
        user_id = user[0]
        cursor.execute("""
            SELECT id, token_hash FROM tokens 
            WHERE user_id = %s AND token_type = 'password_reset' AND expires_at > NOW()
        """, (user_id,))
        token_record = cursor.fetchone()

        if not token_record or not bcrypt.checkpw(code.encode(), token_record[1].encode()):
            return jsonify({"error": "Invalid or expired reset code."}), 400
        
        token_id = token_record[0]
        new_hashed_pw = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()

        cursor.execute("UPDATE users SET password = %s WHERE id = %s", (new_hashed_pw, user_id))
        cursor.execute("DELETE FROM tokens WHERE id = %s", (token_id,))
        
        conn.commit()
        return jsonify({"message": "Password has been reset successfully."})
    except Exception as e:
        conn.rollback()
        logging.error(f"Password reset failed: {e}")
        return jsonify({"error": "An internal error occurred."}), 500

# 👑 Manual superadmin registration (Postman only)
@app.route("/create_superadmin", methods=["POST"])
def create_superadmin():
    try:
        data = request.get_json()
        email = data["email"]
        password = data["password"]
        first_name = data["first_name"]
        last_name = data["last_name"]
        secret_key = data.get("secret_key")

        if secret_key != SUPERADMIN_SECRET_KEY:
            return jsonify({"error": "Invalid secret key"}), 403
        
        if not all([email, password, first_name, last_name]):
            return jsonify({"error": "All fields are required."}), 400

        hashed_pw = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
        # Superadmins are created as verified and active
        cursor.execute("""
            INSERT INTO users (email, password, first_name, last_name, role, state, is_superadmin, is_verified, is_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (email, hashed_pw, first_name, last_name, "superadmin", None, True, True, True))
        conn.commit()
        return jsonify({"message": "Superadmin account created and activated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# 📋 Get pending users (Superadmin only) - Users who have verified email but not activated
@app.route("/pending_users", methods=["GET"])
@superadmin_required
def get_pending_users():
    try:
        cursor.execute("""
            SELECT id, email, first_name, last_name, role, state, created_at FROM users 
            WHERE is_verified = TRUE AND is_active = FALSE AND is_superadmin = FALSE
            ORDER BY created_at DESC
        """)
        users = [{"id": r[0], "email": r[1], "first_name": r[2], "last_name": r[3], "role": r[4], "state": r[5], "created_at": r[6].isoformat()} for r in cursor.fetchall()]
        return jsonify({"pending_users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# 🔓 Activate user (Superadmin only)
@app.route("/activate_user/<int:user_id>", methods=["POST"])
@superadmin_required
def activate_user(user_id):
    try:
        cursor.execute("UPDATE users SET is_active = TRUE WHERE id = %s AND is_superadmin = FALSE", (user_id,))
        if cursor.rowcount == 0:
            return jsonify({"error": "User not found or cannot be activated."}), 404
        conn.commit()
        return jsonify({"message": "User activated successfully"})
    except Exception as e:
        conn.rollback()
        return jsonify({"error": str(e)}), 400

# ... (The rest of your endpoints like deactivate_user, get_all_users, etc., can remain largely the same)
# Make sure to adjust them if the new `is_verified` column impacts their logic.
# For brevity, I'll include the remaining ones without modification unless necessary.

# 🧪 Test auth
@app.route("/me", methods=["GET"])
@token_required
def me():
    try:
        user_id = g.user["user_id"]
        cursor.execute("""SELECT id, email, first_name, last_name, role, state, is_superadmin, is_verified, is_active, created_at FROM users WHERE id = %s""", (user_id,))
        user_data = cursor.fetchone()
        if not user_data:
            return jsonify({"error": "User not found"}), 404
        user_data = {
            "id": user_data[0],
            "email": user_data[1],
            "first_name": user_data[2],
            "last_name": user_data[3],
            "role": user_data[4],
            "state": user_data[5],
            "is_superadmin": user_data[6],
            "is_verified": user_data[7],
            "is_active": user_data[8],
            "created_at": user_data[9].isoformat() if user_data[9] else None
        }
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"user": user_data})

# 🚫 Deactivate user (Superadmin only)
@app.route("/deactivate_user/<int:user_id>", methods=["POST"])
@superadmin_required
def deactivate_user(user_id):
    try:
        cursor.execute("UPDATE users SET is_active = FALSE WHERE id = %s AND is_superadmin = FALSE", (user_id,))
        if cursor.rowcount == 0:
            return jsonify({"error": "User not found or user is a superadmin"}), 404
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
            SELECT id, email, first_name, last_name, role, state, is_verified, is_active, created_at 
            FROM users WHERE is_superadmin = FALSE ORDER BY created_at DESC
        """)
        users = []
        for r in cursor.fetchall():
            users.append({
                "id": r[0], 
                "email": r[1], 
                "first_name": r[2],
                "last_name": r[3],
                "full_name": f"{r[2]} {r[3]}",
                "role": r[4], 
                "state": r[5], 
                "is_verified": r[6], 
                "is_active": r[7], 
                "created_at": r[8].isoformat()
            })
        return jsonify({"users": users})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# 👨‍💼 Get all admins (Superadmin only)
@app.route("/admins", methods=["GET"])
@superadmin_required
def get_all_admins():
    try:
        cursor.execute("""
            SELECT id, email, first_name, last_name, role, state, is_superadmin, is_verified, is_active, created_at 
            FROM users ORDER BY is_superadmin DESC, created_at DESC
        """)
        admins = []
        for r in cursor.fetchall():
            admins.append({
                "id": r[0], 
                "email": r[1], 
                "first_name": r[2],
                "last_name": r[3],
                "full_name": f"{r[2]} {r[3]}",
                "role": r[4], 
                "state": r[5], 
                "is_superadmin": r[6], 
                "is_verified": r[7], 
                "is_active": r[8], 
                "created_at": r[9].isoformat()
            })
        return jsonify({"admins": admins})
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

# 📊 Get upload statistics for all users (Superadmin only)
@app.route("/upload_statistics", methods=["GET"])
@superadmin_required
def get_upload_statistics():
    try:
        # This query uses a window function to rank uploads for each user
        # and selects the top 5 for each. This is more efficient than N+1 queries.
        query = """
            WITH RankedUploads AS (
                SELECT
                    user_id,
                    action_details,
                    created_at,
                    ROW_NUMBER() OVER(PARTITION BY user_id ORDER BY created_at DESC) as rn
                FROM
                    upload_logs
            )
            SELECT
                u.id as user_id,
                u.first_name,
                u.last_name,
                u.email,
                u.state,
                ru.action_details,
                ru.created_at
            FROM
                users u
            LEFT JOIN
                RankedUploads ru ON u.id = ru.user_id
            WHERE
                u.is_superadmin = FALSE
                AND (ru.rn <= 5 OR ru.rn IS NULL)
            ORDER BY
                u.id, ru.created_at DESC;
        """
        cursor.execute(query)
        
        # Group the flat results from the SQL query by user
        user_stats = {}
        for row in cursor.fetchall():
            user_id = row['user_id']
            if user_id not in user_stats:
                user_stats[user_id] = {
                    "user_id": user_id,
                    "name": f"{row['first_name']} {row['last_name']}",
                    "email": row['email'],
                    "state": row['state'],
                    "recent_uploads": []
                }
            
            # Add upload details if they exist (LEFT JOIN can result in NULLs for users with no uploads)
            if row['action_details']:
                user_stats[user_id]['recent_uploads'].append({
                    "details": row['action_details'],
                    "uploaded_at": row['created_at'].isoformat()
                })
        
        # Convert the dictionary of users to a list for the JSON response
        return jsonify({"upload_statistics": list(user_stats.values())})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run
if __name__ == "__main__":
    if not all([DATABASE_URL, SECRET_KEY, SUPERADMIN_SECRET_KEY, EMAIL_HOST, EMAIL_PORT, EMAIL_USER, EMAIL_PASS]):
        logging.warning("One or more critical environment variables are not set. The application might not function correctly.")
    app.run(host="0.0.0.0", port=10000)



