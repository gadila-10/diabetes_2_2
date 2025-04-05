from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
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
    features = [float(request.form[key]) for key in request.form if key != 'gender']
    gender = request.form['gender']
    prediction = model.predict([features])[0]

    if prediction == 1:
        result_text = "You might have Diabetes."
        diet = "Follow a high-fiber, low-carb, low-sugar diet. Include green vegetables, lean protein."
        doctors = "Consult a Diabetologist or Endocrinologist."
    else:
        result_text = "You are not likely diabetic."
        diet = "Maintain a balanced diet with regular exercise."
        doctors = "No urgent consultation needed, but regular checkups are advised."

    return render_template("result.html", result=result_text, diet=diet, doctors=doctors)
