from flask import Flask,request, url_for, redirect, render_template
import pickle
import pandas as pd
import numpy as np
import os 
from PIL import Image
import base64

app = Flask(__name__)
sc=pickle.load(open('mms.pkl','rb'))
model = pickle.load(open('classifier.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/calculator')
def calculator():
    return render_template('calculator.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')


@app.route('/predict',methods=['POST'])
def predict():
   float_features=[float(x) for x in request.form.values()]
   final_features=[np.array(float_features)]
   pred=model.predict(sc.transform(final_features))
   return render_template('result.html',prediction=pred)



if __name__ == '__main__':
    app.run(debug=True)