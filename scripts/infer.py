# class to write metaflow production steps
from metaflow import FlowSpec, step, IncludeFile, environment
import os



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
        print("Starting the flow")
        self.next(self.data_load_flow)

    @step
    def data_load_flow(self):
        from src import DataOps

        print("Loading data")
        self.diabetes = DataOps.feature_engg_class()
        self.data, self.target = self.diabetes.load_data()
        self.next(self.data_process_flow)

    @step
    def data_process_flow(self):
        print("Processing data")
        self.fin_df = self.diabetes.standard_scaling(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.diabetes.split(
            self.fin_df, self.target
        )
        self.next(self.infer_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
                       'EVI_API': os.getenv('EVI_API')
                       })
    @step
    def infer_flow(self):
        from src import ModelOps
        os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY') 
        print("Load the model, make predictions") 
        self.preds=ModelOps.ModelInference().inference(self.X_test)
        self.next(self.end)

    @step
    def monitoring_flow(self):
        from src import ModelOps
        os.environ["EVI_API"] = os.getenv('EVI_API') 
        inference=ModelOps.ModelInference()
        print("Monitoring the model performance")
        ref_dataset_dir=inference.reference_data_download()
        inference.model_monitoring(ref_dataset_dir,self.preds)
        self.next(self.end)
        
    @step
    def end(self):
        print("End of the flow")


if __name__ == "__main__":
    InferFlow()
