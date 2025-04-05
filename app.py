from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    gender = request.form['gender']
    return render_template('form.html', gender=gender)

@app.route('/result', methods=['POST'])
def result():
    gender = request.form['gender']

    # Collect input features in exact model order
    try:
        features = [
            float(request.form['Pregnancies']),
            float(request.form['Glucose']),
            float(request.form['BloodPressure']),
            float(request.form['SkinThickness']),
            float(request.form['Insulin']),
            float(request.form['BMI']),
            float(request.form['DiabetesPedigreeFunction']),
            float(request.form['Age'])
        ]
    except ValueError:
        return "Invalid input. Please make sure all fields are filled correctly."

    prediction = model.predict([features])[0]

    if prediction == 1:
        result_text = "⚠️ You might have Diabetes."
        diet = "Follow a low-carb, high-fiber diet. Avoid sugar, include whole grains, leafy greens, lean proteins."
        doctor = "Consult a Diabetologist or Endocrinologist for further evaluation."
    else:
        result_text = "✅ You are not likely diabetic."
        diet = "Maintain a balanced diet, stay active, and monitor blood sugar levels occasionally."
        doctor = "No urgent need, but routine checkups are recommended."

    return render_template("result.html", result=result_text, diet=diet, doctors=doctor)

if __name__ == "__main__":
    app.run(debug=True)
