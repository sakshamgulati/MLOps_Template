# class to write metaflow production steps
from metaflow import FlowSpec, step, Parameter, IncludeFile,secrets, environment
import os


# @conda_base(python='3.10.1',
#            packages={'scikit-learn': '1.3.2',
#                      'pandas': '2.1.2',
#                      'numpy': '1.26.1',
#                      'python-dotenv': '0.21.1',
#  'confuse':'2.0.1'})
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
        secret_value = os.getenv('DATABASE_PASSWORD')
        print("Secret Value:", secret_value)
        self.fin_df = self.diabetes.standard_scaling(self.data)
        self.X_train, self.X_test, self.y_train, self.y_test = self.diabetes.split(
            self.fin_df, self.target
        )
        self.next(self.ml_flow)

    @environment(vars={'WANDB_API_KEY': os.getenv('WANDB_API_KEY')})
    @step
    def ml_flow(self):
        from src import ModelOps
        import os
        print("Training model") 
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
