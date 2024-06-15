import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os
import yaml
import logging
class ModelInference:
    """
    #docstring for ModelFit class
    #This class is used to train the model and log the metrics to weights and biases

    """

    def __init__(self):

        # Open and read the YAML file
        with open('conf/mlops.yaml', 'r') as file:
            config = yaml.safe_load(file)
        self.model_name = config['model_name']
        self.project_name=config['project_name']
        self.model_type=config['model_type']
        self.wandb_entity=config['wandb_entity']
        
        self.run = wandb.init(project=self.model_name,job_type=self.model_type)
        logging.info(f"Weights and Biases initiated with Run ID: {self.run.id}")
        logging.info(
            f"For more information on the experiments visit: https://wandb.ai/sakshamgulati123"
        )

    def inference(self, X_test):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object
        """
        
        artifact = self.run.use_artifact(f'{self.wandb_entity}/model-registry/{self.model_name}:latest', type='model')
        artifact_dir = artifact.download()
        logging.info(f"Artifact downloaded at: {artifact_dir}")
        logging.info("Model artifact downloaded")
        #load  pickle file
        file_path = os.path.join(artifact_dir, 'model.pickle')  # specify the correct file path
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded from the registry")
        y_pred = model.predict(X_test)
        return y_pred
    
    def monitoring(self):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object
        """
        from evidently.metrics import RegressionQualityMetric, RegressionErrorPlot, RegressionErrorDistribution
        from evidently.metric_preset import DataDriftPreset, RegressionPreset
        from evidently.pipeline.column_mapping import ColumnMapping
        from evidently.report import Report
        target = 'y'
        prediction = 'prediction'
        # numerical_features = ['open', 'high', 'low', 'adjusted close', 'volume', 'dividend amount']
        # categorical_features = ['season', 'holiday', 'workingday', ]#'weathersit']

        column_mapping = ColumnMapping()

        column_mapping.target = target
        column_mapping.prediction = prediction
        # column_mapping.datetime_features = 'ds'
        column_mapping.datetime='ds'
        column_mapping.id = None
                          
        regression_performance_report = Report(metrics=[
            RegressionPreset(),
        ])
        # get the reference dataset
        regression_performance_report.run(reference_data=None, current_data=reference,
                                        column_mapping=column_mapping)
        regression_performance_report
        logging.info("Model performance monitored")
        self.run.finish()
        
        return None
