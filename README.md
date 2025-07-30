# 🧠 AI4Lassa – Machine Learning API for Lassa Fever Risk Prediction

AI4Lassa is a Flask-based REST API that predicts the risk level of Lassa fever based on user-reported symptoms and environmental data. It leverages a trained SVM model and supports integration with mobile and web applications.

---

## 🚀 Features

- Predicts Lassa Fever **risk level** using:
  - Symptoms: Fever, Bleeding, Headache, Vomiting
  - Environmental: Temperature
- Built with **Python**, **Flask**, and **scikit-learn**
- Ready for deployment on **Render**, **Cloud Run**, or any cloud platform
- JSON-based API for seamless mobile/web integration

---

## 🧪 Sample Prediction Payload

Send a POST request to `/predict` with:

```json
{
  "fever": 1,
  "bleeding": 0,
  "headache": 1,
  "vomiting": 1,
  "temperature": 39.2
}
