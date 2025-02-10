from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from flask_cors import CORS

# Load Dataset
df = pd.read_csv("diabetes.csv")

# Prepare Features & Target
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

# Normalize Data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

# Flask App Setup
app = Flask(__name__)
CORS(app)  # Allow frontend to connect

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json["features"]
    data_scaled = scaler.transform([data])  
    prediction = model.predict(data_scaled)[0]
    return jsonify({"prediction": "Diabetic" if prediction == 1 else "Non-Diabetic"})

if __name__ == '__main__':
    app.run(debug=True)
