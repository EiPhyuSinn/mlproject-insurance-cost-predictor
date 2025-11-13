import pandas as pd 
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.logger import get_logger
from src.exception import CustomException
from src.utils import create_directory_if_not_exists,save_object
from src.components.data_transformation import DataTransformationConfig, DataTransformation
from src.components.model_train import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion."""
    file_path: str
    train_path: str = "artifacts/train.csv"
    test_path: str = "artifacts/test.csv"


class DataIngestion:
    """Class for data ingestion."""
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> tuple[str, str]:
        try:
            logger = get_logger()
            logger.info("Starting data ingestion process")

            # Read the dataset
            df = pd.read_csv(self.config.file_path)
            logger.info(f"Dataset read successfully from {self.config.file_path}")

            # Split the dataset into training and testing sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logger.info("Dataset split into training and testing sets")

            create_directory_if_not_exists("artifacts")

            train_set.to_csv(self.config.train_path, index=False)
            logger.info(f"Training set saved to {self.config.train_path}")
            test_set.to_csv(self.config.test_path, index=False)
            logger.info(f"Testing set saved to {self.config.test_path}")
            
            return self.config.train_path, self.config.test_path
        except Exception as e:
            raise CustomException(f"Data ingestion failed: {e}")
        
if __name__ == "__main__":
    config = DataIngestionConfig(file_path="notebook/csv_files/Medical_insurance.csv")
    data_ingestion = DataIngestion(config)
    train_path, test_path = data_ingestion.initiate_data_ingestion()
    print(f"Data ingestion completed. Train path: {train_path}, Test path: {test_path}")

    transformation_config = DataTransformationConfig()
    transformer = DataTransformation(transformation_config)
    train_pkl_path, test_pkl_path = transformer.initiate_data_transformation()
    print(f"Data transformation completed. Train pkl path: {train_pkl_path}, Test pkl path: {test_pkl_path}")

    trainer_config = ModelTrainerConfig(train_pkl_path=train_pkl_path, test_pkl_path=test_pkl_path)
    trainer = ModelTrainer(trainer_config)
    model_path = trainer.initiate_model_trainer()
    print(f"Model training completed. Model saved at: {model_path}")

