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
        self.next(self.ml_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY')})
    @step
    def ml_flow(self):
        from src import ModelOps
        import wandb
        print("Training model")
        os.environ["WANDB_API_KEY"] = os.getenv('WANDB_API_KEY') 
        self.rf_reg = ModelOps.ModelFit()
        self.model = self.rf_reg.model(
            self.X_train, self.X_test, self.y_train, self.y_test
        )
        self.rf_reg.save_model_to_registry(self.model)
        self.next(self.end)

    @step
    def end(self):
        print("End of the flow")


if __name__ == "__main__":
    TrainDeployFlow()
