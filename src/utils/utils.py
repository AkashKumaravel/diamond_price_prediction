import os
import sys
import pandas as pd
import bz2file as bz2
import pickle
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging

def get_numerical_variables():
  data = pd.read_csv('artifact/raw.csv')
  numeric_vars = data.select_dtypes(include=['float64']).columns
  return numeric_vars

def get_categorical_variables():
  data = pd.read_csv('artifact/raw.csv')
  cat_vars = data.select_dtypes(include=['object']).columns
  return cat_vars

def save_object(file_path, obj):
  try:
    dir_name = os.path.dirname(file_path);
    os.makedirs(dir_name, exist_ok = True)

    with bz2.BZ2File(file_path + '.pbz2', 'w') as file_obj:
      pickle.dump(obj, file_obj)

  except Exception as err:
    raise CustomException(err, sys)

def load_object(file_path):
  try:
    data = bz2.BZ2File(file_path + '.pbz2', 'rb')
    return pickle.load(data)
    # with open(file_path, 'rb') as file_obj:
    #   return pickle.load(file_obj)

  except Exception as err:
    raise CustomException(err, sys)

def evaluate_model(x_train, y_train, x_test, y_test, models):
  try:
    report = {}
    logging.info('Model Evaluation Started')
    for x in models:
      curr_model = models[x]
      curr_model.fit(x_train, y_train)

      y_test_pred = curr_model.predict(x_test)

      r2_score_val = r2_score(y_test, y_test_pred)
      report[x] = r2_score_val

    logging.info('Model Evaluation Ended')
    return report

  except Exception as err:
    raise CustomException(err, sys)