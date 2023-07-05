import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_trainer import ModelTrainer,ModelTrainerConfig

class DataIngestion:
    def __init__(self):
        self.raw_data_path: str = os.path.join('artifacts', "data.csv")

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or components")
        try:
            df = pd.read_csv("D:\\MachineLearningProjects\\PROJECT\\OnlinePaymentFraudPrediction\\notebook\\data\\PS_20174392719_1491204439457_log.csv")
            logging.info('Read the ataset as dataframe')

            df = df[["type", "amount", "oldbalanceOrg", "newbalanceOrig","isFraud"]]
            logging.info('Take only those column which are required to for the Model')

            #Create the artifacts directory
            os.makedirs(os.path.dirname(self.raw_data_path),exist_ok=True)

            #Create Raw Data File
            df.to_csv(self.raw_data_path,index=False,header=True)
            logging.info('RawFile Created')


            return (
                self.raw_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)


if __name__ == "__main__":
    obj = DataIngestion()
    rawData = obj.initiate_data_ingestion()

    obj2 = DataTransformation()
    train_Path,test_Path=obj2.encodingOfDataAndTrainTestSplit(rawData)
    train_array,test_array,_= obj2.initiate_data_transformation(train_Path,test_Path)

    obj3 = ModelTrainer()
    obj3.initiate_model_trainer(train_array,test_array)