import os
import sys
import pandas as pd
import numpy as np
import dill

import pickle
from src.exception import CustomException
from sklearn.metrics import r2_score
from src.logger import logging

log=logging.getLogger(__name__)


def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor
    except Exception as e:
        raise CustomException(e,sys)


def save_object(file_path, pre_obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file:
            pickle.dump(pre_obj,file)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report={}

        for i in range(len(list(models))):
            model=list(models.values())[i]

            model.fit(X_train,y_train)

            y_train_hat=model.predict(X_train)

            y_test_hat=model.predict(X_test)

            # train_score=r2_score(y_train,y_train_hat)

            test_score= r2_score(y_test,y_test_hat)

            log.debug(f"{list(models.keys())[i]} : {test_score}")

            report[list(models.keys())[i]]=test_score
        
        return report

    except Exception as e:
        raise CustomException(e,sys)
