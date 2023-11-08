import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle


class ModelFit:
    """
    #docstring for ModelFit class
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):
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

    def save_model(self, reg):
        """
        #docstring for saving the model
        #This function is used to save the model as a pickle file
        #Input: Model object
        #Output: None

        """
        best_model = wandb.Artifact(f"model_{self.run.id}", type="model")

        # Save model as a pickle file
        logging.info(f"Save model as a pickle file")
        with open("artifacts/model1/reg.pkl", "wb") as f:
            pickle.dump(reg, f)
        # model uploaded to model registry at weights and biases
        best_model.add_file("artifacts/model1/reg.pkl")
        self.run.log_artifact(best_model)

        # Link the model to the Model Registry
        self.run.link_artifact(best_model, "model-registry/My Registered Model")

        wandb.finish()

        return None
