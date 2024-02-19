import os
import sys
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor #, GradientBoostingRegressor
from dataclasses import dataclass

from src.utils.constants import DATA_FOLDER_NAME, MODEL_FILE_NAME
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import evaluate_model, save_object

@dataclass
class ModelTrainerConfig:
  model_trainer_path: str = os.path.join(DATA_FOLDER_NAME, MODEL_FILE_NAME)

class ModelTrainer:
  def __init__(self):
    self.model_trainer_config = ModelTrainerConfig()

  def initiate_model_trainer(self, train_arr, test_arr):
    try:
      logging.info('Initiated Model Trainer')
      x_train, y_train = (train_arr[:, :-1], train_arr[:, -1])
      x_test, y_test = (test_arr[:, :-1], test_arr[:, -1])

      models = {
        # "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        # "Gradient Boosting": GradientBoostingRegressor(),
        # "Ridge Regression": Ridge(0.5),
        # "Lasso Regression": Lasso(0.5),
      }

      model_report: dict = evaluate_model(x_train, y_train, x_test, y_test, models)
      best_model_name = max(model_report, key = lambda x: model_report[x])

      best_model = models[best_model_name]
      logging.info('Found the best Model')

      save_object(self.model_trainer_config.model_trainer_path, best_model)
      logging.info('Saved Model Trainer Obj')

      best_r2_score = model_report[best_model_name]
      return best_r2_score

    except Exception as er:
      raise CustomException(er, sys)