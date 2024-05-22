import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import yaml
from pathlib import Path
from prophet import Prophet
from prophet.serialize import model_to_json
import pandas as pd
import numpy as np


class ModelFit:
    """
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):
        with open('conf/mlops.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.model_name = config['model_name']
        self.project_name=config['project_name']
        self.model_type=config['model_type']
        self.model_path=config['saved_model_path']
        
        self.run = wandb.init(project=self.model_name)
        logging.info(f"Weights and Biases initiated with Run ID: {self.run.id}")
        logging.info(
            f"For more information on the experiments visit: https://wandb.ai/sakshamgulati123"
        )

    def model(self, train, test):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object

        """
        model = Prophet()
        model.add_country_holidays(country_name='US')
        model.fit(train)
        close_prices = model.make_future_dataframe(periods=30)
        forecast = model.predict(close_prices)
        forecast=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        merged_df=pd.merge(test, forecast, on='ds', how='inner')
        rms=np.sqrt(np.mean(np.power((np.array(merged_df['y'])-np.array(merged_df['yhat_upper'])),2)))
        logging.info(f"RMS:{rms}")
        wandb.log({"Root Mean squared error": rms})
        return model,forecast
    
    def save_model_to_registry(self,model,wandb=True):
        """
        #docstring for save_model_to_registry
        #This function is used to save the model to the registry
        #Input: None
        #Output: None

        """
        output_dir='artifacts/prophet_model'
        #concatenate output directory with the model name to create a path
        model_path = os.path.join(output_dir, 'serialized_model.json')
        os.makedirs(output_dir, exist_ok=True)
        with open(model_path, 'w') as fout:
            fout.write(model_to_json(model)) 
        
        if wandb:
            registered_model_name = self.model_name
            self.run.link_model(path=model_path, registered_model_name=registered_model_name)
            logging.info("Model saved to weights and biases registry")
        else:
            logging.info("Model not saved to weights and biases registry")
        self.run.finish()
        return None

