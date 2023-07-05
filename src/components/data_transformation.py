import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from sklearn.model_selection import train_test_split

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.train_Path: str = os.path.join('artifacts', "train.csv")
        self.test_Path: str = os.path.join('artifacts', "test.csv")


    def encodingOfDataAndTrainTestSplit(self,rawData):
        try:

            df = pd.read_csv(rawData)
            logging.info('Read the RawData as dataframe')

            df["type"] = df["type"].map({"CASH_OUT": 1, "PAYMENT": 2, "CASH_IN": 3, "TRANSFER": 4, "DEBIT": 5})
            logging.info('OrdinalEncoding of Data Completed')

            self.train_set,self.test_set = train_test_split(df,test_size=0.2,random_state=42)
            logging.info("Train test split Completed")

            #Create&Save Train Data File
            self.train_set.to_csv(self.train_Path,index=False,header=True)
            logging.info('TrainFile Created')

            #Create&Save Train Data File
            self.test_set.to_csv(self.test_Path,index=False,header=True)
            logging.info('TestFile Created')

            return (self.train_Path,
                    self.test_Path)

        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        '''This function si responsible for data trnasformation'''
        try:
            numerical_columns = ["type", "amount", "oldbalanceOrg", "newbalanceOrig"]
            num_pipeline = Pipeline(
                                    steps=[("scaler", StandardScaler())]
                                    )
            logging.info(f"Numerical columns: {numerical_columns}")
            preprocessor = ColumnTransformer([("num_pipeline", num_pipeline, numerical_columns)])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading Train and Test Data completed")

            preprocessing_obj = self.get_data_transformer_object()
            target_column_name = "isFraud"

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )



        except Exception as e:
            raise CustomException(e, sys)
