from flask import Flask,request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import predict
application=Flask(__name__)

app=application

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

app.listen(3000)
if __name__ == '__main__':
    app.run(port=3000)
