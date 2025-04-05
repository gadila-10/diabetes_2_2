from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'secret'
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('gender.html')

@app.route('/select_gender', methods=['POST'])
def select_gender():
    gender = request.form['gender']
    session['gender'] = gender
    return redirect(url_for('input_form'))

@app.route('/input_form')
def input_form():
    gender = session.get('gender', 'Male')
    return render_template('input_form.html', gender=gender)

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    prediction = model.predict([features])[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"

    diet = {
        "Diabetic": "Avoid sugar, eat high-fiber food like whole grains, legumes, vegetables.",
        "Not Diabetic": "Maintain a balanced diet rich in vegetables, fruits, and lean protein."
    }

    doctor = {
        "Diabetic": "Consult a diabetologist or endocrinologist as soon as possible.",
        "Not Diabetic": "No need for doctor consultation, but maintain healthy habits."
    }

    return render_template(
        'result.html',
        prediction=result,
        diet_reco=diet[result],
        doctor_reco=doctor[result]
    )

if __name__ == '__main__':
    app.run(debug=True)
