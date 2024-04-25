import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import yaml
from pathlib import Path


class ModelFit:
    """
    #docstring for ModelFit class
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):
        with open('conf/mlops.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.model_name = config['model_name']
        self.project_name=config['project_name']
        self.model_type=config['model_type']
        self.model_path=config['saved_model_path']
        
        os.environ["WANDB_API_KEY"] = "22787bdec6329d031c43de72471e610b908a8815"
        self.run = wandb.init(project=self.model_name)
        logging.info(f"Weights and Biases initiated with Run ID: {self.run.id}")
        logging.info(
            f"For more information on the experiments visit: https://wandb.ai/sakshamgulati123"
        )

    def model(self, X_train, X_test, y_train, y_test):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object

        """
        reg = RandomForestRegressor().fit(X_train, y_train)
        logging.info("Model Trained")

        wandb.log({"Train R-squared": reg.score(X_train, y_train)})
        logging.info(f"R squared logged:{reg.score(X_train, y_train)}")
        y_pred = reg.predict(X_test)
        assert len(y_pred) == len(y_test), "Length mismatch"

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # logging MSE and R2 score
        logging.info(f"MSE:{mse}")
        logging.info(f"R2:{r2}")
        wandb.log({"Mean squared error": mse, "Test R-squared": r2})
        
        return reg

    def save_model_to_registry(self,model,wandb=True):
        """
        #docstring for save_model_to_registry
        #This function is used to save the model to the registry
        #Input: None
        #Output: None

        """
        filename = Path(self.model_path)
        # Ensure the directory exists
        filename.parent.mkdir(parents=True, exist_ok=True)
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        if wandb:
            registered_model_name = self.model_name
            self.run.link_model(path=filename, registered_model_name=registered_model_name)
            logging.info("Model saved to weights and biases registry")
        else:
            logging.info("Model not saved to weights and biases registry")
        self.run.finish()
        return None

