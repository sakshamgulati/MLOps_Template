# class to write metaflow production steps
from metaflow import FlowSpec, step, Parameter, IncludeFile, environment
import os

class TrainDeployFlow(FlowSpec):
    """
    to run this flow in your local development environment, run the following command:

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
        self.stock_data = DataOps.feature_engg_class()
        self.data=self.stock_data.request_stock_price_hist('AAPL')
        self.stock_data.data=self.data

        self.next(self.data_process_flow)

    @step
    def data_process_flow(self):
        print("Processing data")
        self.train,self.test=self.stock_data.split(use_prophet=True)
        print("Data split successfully")
        self.next(self.ml_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY'),
                       'EVI_API': os.getenv('EVI_API')
                       })
    @step
    def ml_flow(self):
        from src import ModelOps
        print("Training model")
        print(os.getenv('FOO'))
        os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY')
        os.environ["EVI_API"] = os.getenv('EVI_API')  
        self.model=ModelOps.ModelFit()
        #saving reference data for monitoring
        self.model.save_reference_data(self.train)
        #returns a report that can be used to monitor the data quality
        print(self.model.data_quality_check(self.train))
        self.fitted_model,self.forecast=self.model.model(self.train, self.test)
        
        self.model.save_model_to_registry(self.fitted_model)
        self.next(self.end)

    @step
    def end(self):
        print("End of the flow")


if __name__ == "__main__":
    TrainDeployFlow()
