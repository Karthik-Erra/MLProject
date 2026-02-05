import os
import sys
from src.logger import logging
from src.exception import CustomeException
import pandas as pd
import numpy as np
from dataclasses import dataclass
from src.utils import save_object

from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

'''

This class is to create file path for preprocessor file

'''
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer(self):
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = [
                'gender',
                'race_ethnicity',
                'parental_level_of_education',
                'lunch',
                'test_preparation_course'
            ]

            logging.info('Numerical columns tranformation is started')

            num_pipline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaling',StandardScaler())
                ]
            )

            logging.info('Numerical tranformation is completed, categorical tranformation is started')

            cat_pipeline = Pipeline(
                steps = [
                    ('cat_imputer',SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoding',OneHotEncoder()),
                    ('scaling',StandardScaler(with_mean=False))
                ]
            )

            logging.info('Categorical encoding is completed')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipline,numerical_columns),
                    ('cat_pipeline',cat_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomeException(e,sys)
    
    def initiate_data_tranformation(self,train_path,test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')

            logging.info('Obtaining preprocessor object')
            
            preprocessing_obj = self.get_data_transformer()

            target_column = 'math_score'
            numerical_columns = ['writing_score','reading_score']

            input_features_train_df = train_df.drop(columns=target_column)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=target_column)
            target_feature_test_df = test_df[target_column]

            logging.info(f'Applying preprocessing object on training dataframe and testing datframe.')

            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_features_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
                ]

            logging.info('Saved preprocessing object.')
            
            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomeException(e,sys)