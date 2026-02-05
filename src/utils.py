import os
import sys
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomeException
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostRegressor

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            dill.dump(obj,file_obj)

    except Exception as e:
        raise CustomeException(e,sys)
    

def evaluate_model(X_train,y_train,X_test,y_test,models,params):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=params[list(models.keys())[i]]

            if isinstance(model,CatBoostRegressor):
                logging.info('handled catboost model')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)
                report[list(models.keys())[i]] = score
                continue
                
            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)  ## Training the model

            y_train_predict = model.predict(X_train)

            y_test_predict = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_predict)

            test_model_score = r2_score(y_test,y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        logging.info('may be problem in executing model reports in util')
        raise CustomeException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
            
    except Exception as e:
        raise CustomeException(e,sys)