import os
import sys
import pandas as pd

from src.exception import CustomException
# from src.logger import logging
from src.utils.utils import load_object
from src.utils.constants import DATA_FOLDER_NAME, MODEL_FILE_NAME, PREPROCESSOR_FILE_NAME

class PredictPipeline:
  def predict_pipeline(self, features):
    try:
      model_path = os.path.join(DATA_FOLDER_NAME, MODEL_FILE_NAME)
      preprocessor_path = os.path.join(DATA_FOLDER_NAME, PREPROCESSOR_FILE_NAME)

      model = load_object(model_path)
      # logging.info('Loaded Model')

      preprocessor = load_object(preprocessor_path)
      # logging.info('Loaded Preprocessor')

      scaled_data = preprocessor.transform(features)
      # logging.info('Input Data Transformation is Successful')

      prediction = model.predict(scaled_data)
      # logging.info('Price is Predicted Successfully')

      return prediction
    except Exception as err:
      raise CustomException(err, sys)

class CustomData:
  def __init__(self, carat, cut, color, clarity, depth, table, x, y, z):
    self.carat = carat
    self.cut = cut
    self.color = color
    self.clarity = clarity
    self.depth = depth
    self.table = table
    self.x = x
    self.y = y
    self.z = z

  def get_data_as_data_frame(self):
    try:
      input_data = {
        'carat': [self.carat],
        'cut': [self.cut],
        'color': [self.color],
        'clarity': [self.clarity],
        'depth': [self.depth],
        'table': [self.table],
        'x': [self.x],
        'y': [self.y],
        'z': [self.z]
      }

      return pd.DataFrame(input_data)
    except Exception as err:
      raise CustomException(err, sys)