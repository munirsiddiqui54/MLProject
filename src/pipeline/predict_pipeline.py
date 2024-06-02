import pandas as pd
import sys

from src.logger import logging

from sklearn.preprocessing import StandardScaler
from src.utils import load_object
from src.exception import CustomException

log= logging.getLogger(__name__)

def predict(data):
    try:
        df=pd.DataFrame(data)
        preprocessor_path="artifacts\preprocessor.pkl"
        model_path="artifacts\model.pkl"
        preprocessor=load_object(preprocessor_path)
        model=load_object(model_path)

        log.info("Prediction Started...")
        log.info(f"DATA {data}")

        print(df.info())

        scaled_data=preprocessor.transform(df)
        prediction=model.predict(scaled_data)

        log.info(prediction)
        return prediction[0]
    except Exception as e:
        raise CustomException(e,sys)






