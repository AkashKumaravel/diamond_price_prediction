import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.utils.constants import DATA_FOLDER_NAME
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
  train_data_path: str = os.path.join(DATA_FOLDER_NAME, 'train.csv')
  test_data_path: str = os.path.join(DATA_FOLDER_NAME, 'test.csv')
  raw_data_path: str = os.path.join(DATA_FOLDER_NAME, 'raw.csv')

class DataIngestion:
  def __init__(self):
    self.ingestion_config = DataIngestionConfig()

  def initiate_data_ingestion(self):
    try:
      logging.info('Data Ingestion Started')

      data = pd.read_csv('notebook/data/diamond.csv')
      logging.info('Data has been read successfully')

      os.makedirs(DATA_FOLDER_NAME, exist_ok = True)
      logging.info(f'{DATA_FOLDER_NAME} directory has been created')

      data = data.drop(['row_no'], axis = 1)

      data.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

      logging.info('Train Test Split initiated')
      train_set, test_set = train_test_split(data, test_size = 0.2, random_state = 17)

      train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
      test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
      logging.info('Train and Test Data has been ingested successfully')

      return (
        self.ingestion_config.train_data_path,
        self.ingestion_config.test_data_path
      )

    except Exception as err:
      raise CustomException(err, sys)

if __name__ == '__main__':
  obj = DataIngestion()
  train_path, test_path = obj.initiate_data_ingestion()

  data_transformer = DataTransformation()
  train_arr, test_arr, _ = data_transformer.initiate_data_transformation(train_path, test_path)

  model_trainer = ModelTrainer()
  model_trainer.initiate_model_trainer(train_arr, test_arr)