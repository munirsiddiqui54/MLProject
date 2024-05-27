import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info("Data Transformation Started.")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read Train and Test Data Completed in Transformation.")
            logging.info("Obtaining Preprocessor object")
            pre_obj=self.get_data_transformer_object()

            target_col_name="math score"

            print(train_df.head())

            target_feature_train_df=train_df[target_col_name]
            input_feature_train_df=train_df.drop(columns=[target_col_name],axis=1)

            target_feature_test_df=test_df[target_col_name]
            input_feature_test_df=test_df.drop(columns=[target_col_name],axis=1)

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )
            print(input_feature_train_df.columns)


            input_feature_train_arr = pre_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = pre_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                pre_obj=pre_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

    def get_data_transformer_object(self):
        try:
             num_columns=["writing score","reading score"]
             cat_columns=[
                 "gender",
                 "race/ethnicity",
                 "parental level of education",
                 "lunch",
                 "test preparation course"
             ]

             num_pipeline=Pipeline(
                 steps=[
                     # for Missing Values
                     ("imputer",SimpleImputer(strategy="median")),

                     # for scaling
                     ("scalar",StandardScaler(with_mean=False))
                 ]
             )

             cat_pipeline=Pipeline(
                 steps=[
                     ("imputer",SimpleImputer(strategy="most_frequent")),

                     # for cat features 
                     ("one_hot_encoder",OneHotEncoder()),

                     ("scaler",StandardScaler(with_mean=False))
                 ]
             )

             logging.info(f"Numerical Columns {num_columns}")

             logging.info(f"Categorical Columns {cat_columns}")

             preprocessor=ColumnTransformer([
                 ("num_pipeline",num_pipeline,num_columns),
                 ("cat_pipeline",cat_pipeline,cat_columns)
             ])

             return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj =DataIngestion()
    train_path,test_path=obj.initiate_data_ingestion()

    data_transformation= DataTransformation()
    data_transformation.initiate_data_transformation(train_path=train_path,test_path=test_path)
            