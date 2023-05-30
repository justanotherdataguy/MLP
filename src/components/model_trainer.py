import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, eval_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting the training and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'Random Forest': RandomForestRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'Linear Regression': LinearRegression(),
                'K-Neighbors': KNeighborsRegressor(),
                'XGB' : XGBRegressor(),
                'Cat Boost': CatBoostRegressor(verbose=False),
                'ADA Boost' : AdaBoostRegressor()
            }

#This will give us a dictionary of all models with their r2 scores
            model_report:dict=eval_models(X_train= X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)

#Here we are trying to get the best model out of this dictionary
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            if best_model_score<0.7:
                raise CustomException("No Best MOdel")
            
            logging.info("Best Model Found")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            logging.info("models.pkl file saved without error")



            prediction_from_best = best_model.predict(X_test)
            score = r2_score(y_test, prediction_from_best)

            return best_model, score



        except Exception as e:
            raise CustomException(e, sys)
