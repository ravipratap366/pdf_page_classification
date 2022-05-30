from crypt import methods
from flask import Flask,request,jsonify
from app.torch_utils import transformer, get_prediction
#from torch_utils import transformer, get_prediction
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
import sys
import os
import glob
import re
import numpy as np



app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
def allowed_file(filename):
    # xxx.png
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


#/home/ravipartab/Downloads/pytorch_deployment/app/templets/index.html


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        print('hello')

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = get_prediction(file_path,transformer)
        result=preds
        return result
    return None
