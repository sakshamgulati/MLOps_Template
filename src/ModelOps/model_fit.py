import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

class ModelFit:
    """
    #docstring for ModelFit class
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):
        os.environ["WANDB_API_KEY"] = "22787bdec6329d031c43de72471e610b908a8815"
        # os.environ["WANDB_API_KEY"]  = os.getenv('WANDB_API_KEY')
        self.run = wandb.init(project="ml-ops-template")
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
        INIT_MODEL_DIR='init_model'
        
        if wandb:
            
            artifact = wandb.Artifact(name="my_models", type="model")
            model.save(INIT_MODEL_DIR)
            artifact.add_dir(INIT_MODEL_DIR)
            self.run.log_artifact(artifact)
            # artifact.add_dir(local_path="./model.pkl")
            # self.run.log_artifact(artifact)
            logging.info("Model saved to weights and biases registry")
        else:
            logging.info("Model not saved to weights and biases registry")
        self.run.finish()
        return None

