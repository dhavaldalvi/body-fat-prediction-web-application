from flask import Flask, url_for, redirect, render_template, request
import pickle
import numpy as np
import subprocess

app = Flask(__name__)

# Function to run model file 
initialized = False
@app.before_request
def run_model_script():
    global initialized
    if not initialized:
        subprocess.run(['python','model.py'], check=True)
        initialized = True

run_model_script()

# Loading model file
pipeline = pickle.load(open('model.pkl', 'rb'))

# Home page
@app.route('/')
def home():
    return render_template('index.html')

# Result page
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    age = np.array([int(request.form['age'])])
    weight = np.array([float(request.form['weight'])])
    height = np.array([float(request.form['height'])])
    neck = np.array([float(request.form['neck'])])
    chest = np.array([float(request.form['chest'])])
    abdomen = np.array([float(request.form['abdomen'])])
    hip = np.array([float(request.form['hip'])])
    thigh = np.array([float(request.form['thigh'])])
    knee = np.array([float(request.form['knee'])])
    ankle = np.array([float(request.form['ankle'])])
    bicep = np.array([float(request.form['bicep'])])
    forearm = np.array([float(request.form['forearm'])])
    wrist = np.array([float(request.form['wrist'])])
    
    # Making feature array
    features = np.concatenate((age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, bicep, forearm, wrist))
    features = features.reshape(1,13)

    # Prediction
    body_fat_percentage = pipeline.predict(features)[0]

    return render_template('result.html', result=round(body_fat_percentage, 2))

if __name__ == '__main__':
    app.run(debug=True)
