# -*- coding: utf-8 -*-

# Importing libarries
from flask import Flask, render_template, request
import pickle
import numpy as np

# load the Model
filename = 'pima_diabetes-predicion-rf.pkl'
classifier = pickle.load(open(filename, 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['Pregnancies'])
        glucose = int(request.form['Glucose'])
        bp = int(request.form['BloodPressure'])
        st = int(request.form['SkinThickness'])
        insulin = int(request.form['Insulin'])
        bmi = float(request.form['BMI'])
        dpf = float(request.form['DiabetesPedigreeFunction'])
        age = int(request.form['Age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        
        return render_template('result.html', prediction=my_prediction)
    
    
if __name__ == '__main__':
    app.run(debug=True)