import pandas as pd 
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object
import numpy as np 

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest Regressor": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(), 
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor()
}


def evaluate(test,pred):
    r2 = r2_score(test,pred)
    mae = mean_absolute_error(test,pred)
    rmse = np.sqrt(mean_squared_error(test,pred))
    return r2,mae,rmse


@dataclass
class ModelTrainerConfig:
    train_pkl_path: str = "artifacts/train.pkl"
    test_pkl_path: str = "artifacts/test.pkl"
    trained_model_path: str = "artifacts/model.pkl"

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def initiate_model_trainer(self) -> str:
        try:
            logger = get_logger()
            logger.info("Starting model training process")

            # Load transformed training and testing data
            train_array = pd.read_pickle(self.config.train_pkl_path)
            test_array = pd.read_pickle(self.config.test_pkl_path)

            X_train = train_array[:,:-1]
            y_train = train_array[:,-1]

            X_test = test_array[:,:-1]
            y_test = test_array[:,-1]

            best_model_name = None
            best_model_score = -np.inf
            best_model = None

            for model_name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2, mae, rmse = evaluate(y_test, y_pred)

                logger.info(f"{model_name} -- R2: {r2}, MAE: {mae}, RMSE: {rmse}")

                if r2 > best_model_score:
                    best_model_score = r2
                    best_model_name = model_name
                    best_model = model

            logger.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")

            save_object(self.config.trained_model_path, best_model)
            logger.info(f"Trained model saved at {self.config.trained_model_path}")

            return self.config.trained_model_path

        except Exception as e:
            raise CustomException(f"Model training failed: {e}")