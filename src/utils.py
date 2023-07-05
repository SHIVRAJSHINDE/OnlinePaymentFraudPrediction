import os
import sys
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


import numpy as np
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)


def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            print(model)

            para = param[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)
            X_train_prediction = model.predict(X_train)
            training_data_accuracy = accuracy_score(y_train, X_train_prediction)
            print("training_data_accuracy = ", training_data_accuracy)
            X_test_prediction = model.predict(X_test)
            test_data_accuracy = accuracy_score(y_test, X_test_prediction)
            print("test_data_accuracy = ", test_data_accuracy)
            precision_train = precision_score(y_train, X_train_prediction)
            print('Training data Precision = ', precision_train)
            precision_test = precision_score(y_test, X_test_prediction)
            print('Test data Precision = ', precision_test)
            recall_train = recall_score(y_train, X_train_prediction)
            print('Training data Recall = ', recall_train)
            recall_test = recall_score(y_test, X_test_prediction)
            print('Test data Recall = ', recall_test)
            report[list(models.keys())[i]] = precision_test

        return report

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)


