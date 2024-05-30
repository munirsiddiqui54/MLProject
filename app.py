from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import predict
application=Flask(__name__)

app=application


CORS(app)

@app.route('/')
def index():
    return render_template('index.html',prediction=None)


@app.route('/submit', methods=['POST'])
def submit():
    data =   {
        "gender": [request.form['gender']],
        "race/ethnicity": [request.form['race_ethnicity']],
        "parental level of education":[ request.form['parental_education']],
        "lunch":[ request.form['lunch']],
        "test preparation course":[ request.form['test_preparation']],
        # "math score": [request.form['math_score']],
        "reading score": [request.form['reading_score']],
        "writing score": [request.form['writing_score']]
    }
    # For now, just print the data
    prediction= predict(data)
    
    # Here, you can process the data or save it as needed
    return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run()
