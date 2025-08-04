🧠 AI4Lassa API

An intelligent web API for predicting Lassa fever cases using machine learning and for managing statistical health data across different states in Nigeria. Built with Flask, PostgreSQL, and deployed on Render.

🚀 Features
🤖 ML Prediction using a trained SVM model
🔐 Secure JWT-based authentication
👤 Superadmin and Admin roles
📊 Upload and retrieve Lassa fever statistics
📥 Bulk upload support
🌍 PostgreSQL cloud database integration

📂 Endpoints
🔐 Authentication
1. POST /register
Register a user (superadmin or state admin).

json
{
  "email": "admin@example.com",
  "password": "yourpassword",
  "role": "admin",          // or "superadmin"
  "state": "Lagos"          // required only for admin
}

2. POST /login
Login and receive a JWT token.

json
{
  "email": "admin@example.com",
  "password": "yourpassword"
  }
  Returns:
  json
{
  "token": "your.jwt.token"
}

🧪 Machine Learning Prediction
3. POST /predict
Make a Lassa fever prediction from symptoms.

json
{
  "fever": 1,
  "bleeding": 0,
  "headache": 1,
  "vomiting": 0,
  "temperature": 38.2
}
Returns:

json
{
  "prediction_percentage": "76.45%"
}

📥 Upload Stats
POST /upload_stats
Upload monthly stats. (Authenticated)

superadmin can upload for any state
admin can upload only for their state
Single or bulk upload format:

json

[
  {
    "state": "Lagos",
    "year": 2024,
    "month": 1,
    "cases": 30,
    "deaths": 2,
    "recoveries": 25
  },
  {
    "state": "Lagos",
    "year": 2024,
    "month": 2,
    "cases": 45,
    "deaths": 5,
    "recoveries": 30
  }
]
Headers:
Authorization: Bearer <your-token>

👁️ View History
GET /history?state=Lagos&start_year=2023&end_year=2024&start_month=1&end_month=12
Returns:

json

[
  {
    "year": 2024,
    "month": 1,
    "cases": 30,
    "deaths": 2,
    "recoveries": 25
  },
  ...
]
🛠 Create Tables (One-time use)
GET /create_table
Creates the lassa_stats and users tables in the database.

🧑‍💻 Local Setup
🔧 Requirements
Python 3.9+
psycopg2, flask, joblib, bcrypt, python-jose, numpy, etc.
PostgreSQL (cloud or local)
Render account (for deployment)

🛠 Environment Variables
Set these on Render or in a .env file:

DATABASE_URL=your_postgres_connection_string
SECRET_KEY=your_secret_key

▶ Run locally
bash
Copy
Edit
pip install -r requirements.txt
python app.py
🧠 Model
SVM model trained for Lassa fever prediction.
Scaler and model are loaded from scaler.pkl and ai4lassa_svm_model.pkl.

🔐 Roles
Role	Access Rights
Superadmin	Can register admins and upload stats for all states
Admin	Can upload stats only for their assigned state

📤 Deployment
Deployed on Render using:
Web Service with Python environment
Connected GitHub repo
Environment variables set on Render dashboard

📬 Contact
Built by Samuel Akinwumi Adeniji
Email: ifeoluwasamuel40@gmail.com
