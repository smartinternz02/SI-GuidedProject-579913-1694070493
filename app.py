import pickle
from flask import Flask, render_template, request
import pandas as pd
import numpy as np

app = Flask(__name__,static_folder="Static")
model = pickle.load(open('C:/Users/admin/Desktop/blood donation/model1.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about') # rendering the html template
def about():
    return render_template('about.html')

@app.route('/findthedonor') # rendering the html template
def findthedonor():
    return render_template('findthedonor.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    recency= float(request.form['recency'])
    frequency= float(request.form['frequency'])
    monetary= float(request.form['monetary'])
    time= float(request.form['time'])
    #int_features = [int(x) for x in request.form.values()]
    features = np.array([recency, frequency, monetary, time])
    features_2d = features.reshape(1, -1)
    prediction = model.predict(features_2d)
    output = prediction[0]
    return render_template('result.html', prediction_text='chance of donor to donate blood is{}'.format(output))

if __name__=="__main__":
   app.run(debug=True)