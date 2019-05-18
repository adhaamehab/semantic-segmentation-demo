# -*- coding: utf-8 -*-
from src.model import DeepLabModel
from src.utils import inference
from flask import Flask, send_file, redirect, url_for, render_template, request





app = Flask(__name__, template_folder='templates')

@app.route('/predict')
def predict():
    url = request.args.get('img')    
    result = inference(MODEL, url)
    return send_file(result,
                     attachment_filename='segmantation.png',
                     mimetype='image/png')

@app.route('/')
def index():
    return render_template('main.html')



if __name__ == "__main__":
    MODEL = DeepLabModel('./models/ade20kmode_may2019.gz')
    app.run('0.0.0.0', port=3030)
