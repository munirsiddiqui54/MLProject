import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_model

log=logging.getLogger(__name__)

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_arr,test_arr):
        try:
            log.info("Spliting Train array and Test Array")
            X_train,y_train,X_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1], 
                test_arr[:,:-1],
                test_arr[:,-1]
            )

            models={
                "RandomForest":RandomForestRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
                "LinearRegression":LinearRegression(),
                "GradientBoosting":GradientBoostingRegressor(),
                "K-NeighboursRegressor":KNeighborsRegressor(),
                "XGBRegressor":XGBRegressor(),
                # "CatBoostingRegressor":CatBoostRegressor(verbose=False),
                "AdaBoostRegressor":AdaBoostRegressor()
            }

            model_report :dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)
            # Best score of models among all:
            best_score=max(sorted(model_report.values()))


            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_score)
            ]


            if best_score<0.6:
                raise CustomException("No Best Model Found")
            
            best_model=models[best_model_name]

            log.info(f"Best Model found: {best_model_name}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                pre_obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_s=r2_score(y_test,predicted)
            return r2_s


            

        except Exception as e:
            raise CustomException(e,sys)

