from flask import Flask,request, render_template
import numpy as np
import pandas as pd
from flask_cors import CORS

import sys
from src.utils import load_object
import pickle
from src.exception import CustomException

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from sklearn.preprocessing import StandardScaler
# from src.pipeline.predict_pipeline import predict
application=Flask(__name__)

app=application




obj =DataIngestion()
train_path,test_path=obj.initiate_data_ingestion()

data_transformation= DataTransformation()
train_ar,test_ar ,_=data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)

model_trainer=ModelTrainer()
score=model_trainer.initiate_model_trainer(train_arr=train_ar,test_arr=test_ar)
print(score)

preprocessor_path="artifacts/preprocessor.pkl"
model_path="artifacts/model.pkl"
preprocessor=load_object(preprocessor_path)
model=load_object(model_path)


CORS(app)

@app.route('/')
def index():
    return render_template('index.html',prediction=None)



def predict(data):
    try:
        df=pd.DataFrame(data)



        print(df.info())
        scaled_data=preprocessor.transform(df)
        prediction=model.predict(scaled_data)

        # log.info(prediction)
        return prediction[0]
    except Exception as e:
        raise CustomException(e,sys)

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
