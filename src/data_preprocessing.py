import pandas as pd
from src.logger import Logger


class DataPreprocessor:

    def __init__(self, file_path):
        self.file_path = file_path
        self.logger = Logger()

    def load_data(self):
        try:
            self.logger.info("Loading heart disease dataset")

            data = pd.read_csv(self.file_path)

            self.logger.info("Dataset loaded successfully")
            self.logger.info(f"Dataset shape: {data.shape}")

            return data

        except FileNotFoundError:
            self.logger.error("Dataset file not found")
            print("Error: heart.csv file was not found.")

        except Exception as e:
            self.logger.error(f"Error while loading dataset: {e}")
            print(f"Error: {e}")

        return None

    def check_data(self, data):

        try:
            if data is None:
                raise ValueError("Dataset is empty or could not be loaded.")

            print("\nFirst 5 rows:")
            print(data.head())

            print("\nDataset Shape:")
            print(data.shape)

            print("\nColumn Names:")
            print(data.columns.tolist())

            print("\nMissing Values:")
            print(data.isnull().sum())

            print("\nData Types:")
            print(data.dtypes)

            self.logger.info("Dataset validation completed")

        except Exception as e:
            self.logger.error(f"Error while checking dataset: {e}")
            print(f"Error: {e}")