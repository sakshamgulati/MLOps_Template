# class to write metaflow production steps
from metaflow import FlowSpec, step, IncludeFile, environment, schedule, conda_base
import os
from datetime import datetime
from src.DataOps import feature_engg_class
from src.ModelOps import ModelInference
import pandas as pd

@schedule(daily=True)
# @conda_base(python='3.10', libraries={'wandb': '0.16.6', 'pandas': '2.2.2', 'scikit-learn': '0.24.2', 'prophet':'1.1.5'})
class InferFlow(FlowSpec):
    """
    This class is used run inference from a trained model in a model repository.
    We have used Weights and Biases for tracking the model performance and to save the model in their free registry
    
    
    to run this flow in your local development environment, run the following command:
    pip install metaflow
    python scripts/infer.py run
    pre-requisites:
    1. Ensure that WANDB_API_KEY is set in your environment variables
    
    """

    # include the conf/config.yaml file in includefile
    includefile = IncludeFile(
        "configfile", help="Include the config file", default="conf/mlops.yaml"
    )

    @step
    def start(self):
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print('time is %s' % now)
        print("Starting the flow")
        from dotenv import load_dotenv
        load_dotenv()
        self.next(self.data_load_flow)

    @step
    def data_load_flow(self):
        print("Loading data")
        self.stock_data = feature_engg_class()
        self.data=self.stock_data.request_stock_price_hist('AAPL')
        self.stock_data.data=self.data.tail(5)

        self.next(self.feature_engg_flow)
    
    @step
    def feature_engg_flow(self):
        print("Feature Engineering")
        self.stock_data.data = self.stock_data.data.reset_index()
        self.stock_data.data=self.stock_data.data[['date','close']]
        self.stock_data.data.columns = ['ds','y']
        self.stock_data.data['ds'] = pd.to_datetime(self.stock_data.data['ds'])
        self.stock_data.data['y'] = self.stock_data.data['y'].astype(float)
        print(self.stock_data.data.info())
        self.next(self.infer_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
                       'EVI_API': os.getenv('EVI_API')
                       })
    @step
    def infer_flow(self):
        
        os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY') 
        print("Load the model, make predictions")
        self.inference=ModelInference() 
        self.preds=ModelInference().inference(self.stock_data.data)
        self.next(self.monitoring_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
                       'EVI_API': os.getenv('EVI_API')
                       })
    @step
    def monitoring_flow(self):
        os.environ["EVI_API"] = os.getenv('EVI_API')
        inference=ModelInference()
        print("Monitoring the model performance")
        ref_dataset_dir=inference.reference_data_download()
        print("Artifact dataset downloaded successfully")
        inference.model_monitoring(ref_dataset_dir,self.preds)
        self.next(self.end)
        
    @step
    def end(self):
        print("End of the flow")


if __name__ == "__main__":
    InferFlow()
