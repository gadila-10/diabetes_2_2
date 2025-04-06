from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        gender = request.form.get('gender', 'unknown')
        return render_template('form.html', gender=gender)
    else:
        return render_template('index.html')  # redirect if accessed directly

@app.route('/result', methods=['POST'])
def result():
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
        return "❌ Invalid input! Please enter numeric values only."

    prediction = model.predict([features])[0]

    if prediction == 1:
        message = "⚠️ You might have Diabetes."
        diet = "Low-sugar, high-fiber meals. Avoid processed carbs."
        advice = "Consult a Diabetologist immediately."
    else:
        message = "✅ You are not diabetic."
        diet = "Maintain healthy meals and regular exercise."
        advice = "Routine check-ups recommended."

    return render_template("result.html", result=message, diet=diet, doctors=advice)

# Optional: handle 404 errors
@app.errorhandler(404)
def page_not_found(e):
    return "Page not found. Go back to the home page.", 404

if __name__ == '__main__':
    app.run(debug=True)
