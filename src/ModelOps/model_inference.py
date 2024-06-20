import logging
from sklearn.linear_model import LinearRegression
import wandb
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import os
import yaml
import logging
from prophet.serialize import  model_from_json

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
        file_path = os.path.join(artifact_dir, 'serialized_model.json')  # specify the correct file path
        with open(file_path, 'r') as fin:
            model = model_from_json(fin.read())  # Load model
        logging.info("Model loaded from the registry")
        y_pred = model.predict(X_test)
        return y_pred
    
    def reference_data_download(self):
        """
        #docstring for model
        #This function is used to download the refernece data from the weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object
        """
        try:
            ref_dataset = self.run.use_artifact(f'{self.wandb_entity}/{self.model_name}/reference-dataset:latest', type='dataset')
            ref_dataset_dir = ref_dataset.download()
            logging.info(f"Artifact downloaded at: {ref_dataset_dir}")
            logging.info("Model artifact downloaded")
            return ref_dataset_dir
        except:
            logging.error("Reference dataset not found")
            return None
        
    def model_monitoring(self,ref_dataset_dir,preds):
        """
        #docstring for model
        #This function is used to train the model and log the metrics to weights and biases
        #Input: X_train, X_test, y_train, y_test
        #Output: Model object
        """
        from evidently.metric_preset import RegressionPreset
        from evidently.pipeline.column_mapping import ColumnMapping
        from evidently.report import Report
        from evidently.ui.workspace.cloud import CloudWorkspace

        ws = CloudWorkspace(
        token=os.getenv('EVI_API'),
        url="https://app.evidently.cloud")
        target = 'y'
        prediction = 'prediction'

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
        reference_data = pd.read_csv(os.path.join(ref_dataset_dir,"reference_data.csv"))
        regression_performance_report.run(reference_data=reference_data, current_data=preds,
                                        column_mapping=column_mapping)
        os.makedirs("artifacts/model_quality",exist_ok=True)
        regression_performance_report.save("artifacts/model_quality/regression_performance_report.json")
        #TODO: extract the same project id used in training and save the report in the same project
        ws.add_report(self.project.id, regression_performance_report)
        self.run.finish()
        
        return None
