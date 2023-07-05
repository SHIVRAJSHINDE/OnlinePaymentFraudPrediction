import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'proprocessor.pkl')
            print("Before Loading")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)



class CustomData:
    def __init__(self,
        type : str,
        amount : int,
        oldbalanceOrg : int,
        newbalanceOrig : int):

        self.type = type
        self.amount = amount
        self.oldbalanceOrg = oldbalanceOrg
        self.newbalanceOrig = newbalanceOrig

    def get_data_as_data_frame(self):
        try:
            if self.type == "CASH_OUT":
                self.type = 1
            elif self.type == "PAYMENT":
                self.type = 2
            elif self.type == "CASH_IN":
                self.type = 3
            elif self.type == "TRANSFER":
                self.type = 4
            elif self.type == "DEBIT":
                self.type = 5

            custom_data_input_dict = {
            "type":[self.type],
            "amount" :[self.amount],
            "oldbalanceOrg" :[self.oldbalanceOrg],
            "newbalanceOrig" :[self.newbalanceOrig]

            }

            return  pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
