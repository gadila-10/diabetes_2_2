#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('diabetes_model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['POST'])
def form():
    gender = request.form['gender']
    return render_template('form.html', gender=gender)

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
        return "Invalid input! Please enter numeric values."

    prediction = model.predict([features])[0]

    if prediction == 1:
        message = "⚠️ You might have Diabetes."
        diet = "Low-sugar, high-fiber meals. Avoid processed carbs."
        advice = "Consult a Diabetologist immediately."
    else:
        message = "✅ You are not diabetic."
        diet = "Maintain healthy meals and exercise."
        advice = "Routine check-ups recommended."

    return render_template("result.html", result=message, diet=diet, doctors=advice)

if __name__ == '__main__':
    app.run(debug=True)


# In[ ]:




