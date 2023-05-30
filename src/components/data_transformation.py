import sys
from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transform_object(self):
        '''This function is for preprocessing our data and making it ready for modelling'''
        try:
            numerical_columns = ['writing_score', 'reading_score']
            categorical_columns = [
                'gender', 'race_ethnicity', 'parental_level_of_education',
                'lunch', 'test_preparation_course'
            ]

            num_pipeline = Pipeline(steps = [('num_imp', SimpleImputer(strategy='median')),
                                              ('num_scaler', StandardScaler())])
            
            cat_pipeline = Pipeline(steps= [('cat_imp', SimpleImputer(strategy="most_frequent")),
                                            ('cat_ohe', OneHotEncoder()),
                                            ('cat_sca', StandardScaler(with_mean=False))])
            
            logging.info("Numerical and Categorical Pipeline Created")
            
            preprocessor = ColumnTransformer(
                [('num_pipeline', num_pipeline, numerical_columns),
                 ('cat_pipeline', cat_pipeline, categorical_columns)
                 ]
            )

            return preprocessor


        except Exception as e:
            raise CustomException(e, sys)
        

    def initiate_data_transformation(self, train_path, test_path):

        logging.info("Inititating Data Transformation")

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read test and train data successfully")

            logging.info("Getting Preprocessor")

            preprocessing_obj = self.get_data_transform_object()

            target_column_name = ['math_score']
            numerical_columns = ['writing_score', 'reading_score']

            input_feature_train_df = train_df.drop(columns=target_column_name, axis=1)
            target_feature_train_df = train_df['math_score']

            input_feature_test_df = test_df.drop(columns=target_column_name, axis=1)
            target_feature_test_df = test_df['math_score']

            logging.info("Starting Preprocessing on the modified train and test df")

            input_feat_preprocessed_train = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feat_preprocessed_test = preprocessing_obj.fit_transform(input_feature_test_df)

        #Making the train and test data arrays after preprocessing
            train_arr = np.c_[input_feat_preprocessed_train, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feat_preprocessed_test, np.array(target_feature_test_df)]

            logging.info("Saving Preprocessing Object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj= preprocessing_obj
            )
            logging.info("Saved Preprocessing Object")


            return (
                train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path
            )
        
        


        except Exception as e:
            raise CustomException(e,sys)
        