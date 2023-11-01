#class to write metaflow production steps
from metaflow import FlowSpec,step,Parameter,IncludeFile


# @conda_base(python='3.10.1',
#            packages={'scikit-learn': '1.3.2',
#                      'pandas': '2.1.2',
#                      'numpy': '1.26.1',
#                      'python-dotenv': '0.21.1',
                    #  'confuse':'2.0.1'})
class MyFlow(FlowSpec):
    #include the conf/config.yaml file in includefile
    includefile=IncludeFile('configfile',help='Include the config file',default='conf/smartprice.yaml')
    @step
    def start(self):
        print('Starting the flow') 
        self.next(self.data_load_flow)
    
    @step
    def data_load_flow(self):
        from src import DataOps
        print('Loading data')
        self.feat=DataOps.feature_engg_class()
        self.num_df,self.cat_df,self.output=self.feat.load_data()
        self.next(self.data_process_flow)

    @step
    def data_process_flow(self):
        print('Processing data')
        self.fin_df=self.feat.min_max_scaling(self.cat_df,self.num_df)
        self.X_train,self.X_test,self.y_train,self.y_test=self.feat.split(self.fin_df,self.output)
        self.next(self.ml_flow)
    
    @step
    def ml_flow(self):
        from src import ModelOps
        print('Training model')
        l_reg=ModelOps.model_fit()
        self.metric=l_reg.model(self.X_train,self.X_test,self.y_train,self.y_test)
        self.next(self.publishing_api)
    
    @step
    def publishing_api(self):
        print('publishing of the results')
        self.next(self.end)
    
    @step
    def end(self):
        print('End of the flow')

if __name__=='__main__':
    MyFlow()