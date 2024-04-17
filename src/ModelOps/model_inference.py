import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

class ModelInference:
    """
    #docstring for ModelFit class
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):
        os.environ["WANDB_API_KEY"] = "22787bdec6329d031c43de72471e610b908a8815"
        # os.environ["WANDB_API_KEY"]  = os.getenv('WANDB_API_KEY')
        self.run = wandb.init(project="ml-ops-template",job_type="inference")
        logging.info(f"Weights and Biases initiated with Run ID: {self.run.id}")
        logging.info(
            f"For more information on the experiments visit: https://wandb.ai/sakshamgulati123"
        )

    def inference(self, y_test):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object

        """
        artifact = self.run.use_artifact('sakshamgulati123/ml-ops-template/model_atzgbhx7:v0', type='model')
        artifact_dir = artifact.download()
        logging.info(f"Artifact downloaded at: {artifact_dir}")
        logging.info("Model artifact downloaded")
        #load  pickle file
        file_path = os.path.join(artifact_dir, 'reg.pkl')  # specify the correct file path
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded from the registry")
        y_pred = model.predict(y_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # logging MSE and R2 score
        logging.info(f"MSE:{mse}")
        logging.info(f"R2:{r2}")
        wandb.log({"Mean squared error": mse, "Test R-squared": r2})
        
        return y_pred
