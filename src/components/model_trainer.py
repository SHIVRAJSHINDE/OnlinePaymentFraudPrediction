import os
import sys
import numpy as np

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from dataclasses import dataclass


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test data")
            X_train,y_train,X_test,y_test =(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "DeciTree": DecisionTreeClassifier(),
                "Logistic Regression": LogisticRegression(),
            }
            param = {
                "Logistic Regression": {
                    'penalty': ['l2'],
                    'C': np.logspace(-4, 4, 20),
                    'solver': ['lbfgs'],
                    'max_iter': [100, 1000]
                },
                "DeciTree":
                    {
                        'max_depth': [3, 5, 7],
                        'min_samples_split': [2, ],
                        'min_samples_leaf': [1, 2, 4],
                    }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test = X_test,y_test = y_test, models=models,param=param)

            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))


            ## Get best model from the dict
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model Found")

            logging.info("process completed")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                        )

            predicted = best_model.predict(X_test)

        except Exception as e:
            raise CustomException(e, sys)
