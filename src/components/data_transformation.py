import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.constants import DATA_FOLDER_NAME, TARGET_COLUMN, PREPROCESSOR_FILE_NAME
from src.exception import CustomException
from src.logger import logging
from src.utils.utils import get_numerical_variables, get_categorical_variables, save_object

@dataclass
class DataTransformationConfig:
  pre_processor_obj: str = os.path.join(DATA_FOLDER_NAME, PREPROCESSOR_FILE_NAME)

class DataTransformation:
  def __init__(self):
    self.data_transformation_config = DataTransformationConfig()

  def get_data_transformer_obj(self):
    try:
      numerical_variables = get_numerical_variables()
      categorical_variables = get_categorical_variables()

      num_pipeline = Pipeline(
        steps = [
          ('imputer', SimpleImputer(strategy = 'median')),
          ('scaler', StandardScaler(with_mean = False))
        ]
      )

      categorical_pipeline = Pipeline(
        steps = [
          ('imputer', SimpleImputer(strategy = 'most_frequent')),
          ('encoder', OneHotEncoder())
        ]
      )

      preprocessor = ColumnTransformer(
        [
          ('Numerical Pipeline', num_pipeline, numerical_variables),
          ('Categorical Pipeline', categorical_pipeline, categorical_variables)
        ]
      )

      return preprocessor
    except Exception as err:
      raise CustomException(err, sys)

  def initiate_data_transformation(self, train_path, test_path):
    try:
      logging.info('Data Transformation Started')
      train_set = pd.read_csv(train_path)
      test_set = pd.read_csv(test_path)

      pre_processing_obj = self.get_data_transformer_obj()

      input_col_train = train_set.drop([TARGET_COLUMN], axis = 1);
      target_col_train = train_set[TARGET_COLUMN]

      input_col_test = test_set.drop([TARGET_COLUMN], axis = 1);
      target_col_test = test_set[TARGET_COLUMN]

      logging.info('Applying preprocessing on train and test dataset')
      input_train_arr = pre_processing_obj.fit_transform(input_col_train)
      input_test_arr = pre_processing_obj.transform(input_col_test)

      train_arr = np.c_[input_train_arr, np.array(target_col_train)]
      test_arr = np.c_[input_test_arr, np.array(target_col_test)]
      logging.info('Saved Preprocessing Obj')

      save_object(self.data_transformation_config.pre_processor_obj, pre_processing_obj)
      return (train_arr, test_arr, self.data_transformation_config.pre_processor_obj)
    except Exception as err:
      raise CustomException(err, sys)