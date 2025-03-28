from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("diabetes_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route('/')
def home():
    return render_template('index.html')  # Welcome page

@app.route('/predict', methods=['POST'])
def predict():
    # Get form values
    features = [float(request.form[feature]) for feature in ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']]
    
    # Preprocess input
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    
    # Provide recommendations
    diet = "Follow a balanced low-carb diet with high fiber." if prediction == 1 else "Maintain a healthy diet to prevent diabetes."
    doctor = "Consult a doctor for further evaluation." if prediction == 1 else "Regular health check-ups are recommended."
    
    return render_template('result.html', result=result, diet=diet, doctor=doctor)

if __name__ == '__main__':
    app.run(debug=True)
