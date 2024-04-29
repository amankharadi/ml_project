import os
import sys
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomeException
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from dataclasses import dataclass
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    prepocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_onject(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = ["gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
                ]
            
            num_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoded',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))

                ]
            )
            logging.info("Numerical feature scaling with Standerscaler")
            logging.info("Categorical feature done encoding and Standerscaler")

            preprocessor =ColumnTransformer([
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipeline",cat_pipeline,categorical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomeException(e,sys)

    def initite_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test data Complete")

            preporcessor_obj = self.get_data_transformation_onject()

            traget_feature_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            input_feature_train_df = train_df.drop(columns=[traget_feature_name])
            traget_feature_train_df = train_df[traget_feature_name]

            input_feature_test_df = test_df.drop(columns=[traget_feature_name])
            traget_feature_test_df = test_df[traget_feature_name]

            logging.info(f"Apply preprocessing on Train and Test DataFrame")

            input_feature_train_arr = preporcessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preporcessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr,np.array(traget_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(traget_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(
                file_path =self.data_transformation_config.prepocessor_obj_file_path,
                obj = preporcessor_obj
            )
            
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.prepocessor_obj_file_path,
            )




        except Exception as e:
            raise CustomeException(e,sys)