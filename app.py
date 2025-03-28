#!/usr/bin/env python
# coding: utf-8

# In[2]:


from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load the trained ML model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

users = {"admin": "password"}  # Simple user authentication dictionary

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username] == password:
            session['username'] = username
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials"
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return "Registration is disabled in this version."

@app.route('/dashboard')
def dashboard():
    if 'username' in session:
        return render_template('dashboard.html', username=session['username'])
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])
    prediction = model.predict(input_data)
    result = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
    
    if prediction[0] == 1:
        diet = "Follow a low-carb diet with high fiber and protein intake. Avoid sugar and processed foods."
        consultation = "Consult an endocrinologist for proper diabetes management."
    else:
        diet = "Maintain a balanced diet with healthy carbs, proteins, and fats. Regular exercise is recommended."
        consultation = "Regular checkups with a physician are recommended."
    
    return render_template('result.html', result=result, diet=diet, consultation=consultation)

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)

