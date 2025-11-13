import pandas as pd 
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object


class DataTransformationConfig:
    """Configuration for data transformation."""
    def __init__(self):
        self.transformed_train_path = "artifacts/train.csv"
        self.transformed_test_path = "artifacts/test.csv"   
        self.train_pkl_path = "artifacts/train_set.pkl"
        self.test_pkl_path = "artifacts/test_set.pkl"

class DataTransformation:
    """Class for data transformation."""
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(self) -> tuple[str, str]:  
        train_df = pd.read_csv(self.config.transformed_train_path)
        test_df  = pd.read_csv(self.config.transformed_test_path)

        numerical_features = train_df.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = train_df.select_dtypes(include=['object']).columns
        logger = get_logger()
        try:
            logger.info("Starting data transformation process")

            numerical_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])

            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numerical_transformer, numerical_features),
                    ('cat', categorical_transformer, categorical_features)
                ])

            X_train = preprocessor.fit_transform(train_df)
            X_test = preprocessor.transform(test_df)

            save_object(file_path=self.config.train_pkl_path, obj=X_train)
            logger.info(f"Transformed training set saved to {self.config.train_pkl_path})")
            save_object(file_path=self.config.test_pkl_path, obj=X_test)
            logger.info(f"Transformed testing set saved to {self.config.test_pkl_path})")

            return self.config.train_pkl_path, self.config.test_pkl_path
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise CustomException(f"Data transformation failed: {e}")
        





