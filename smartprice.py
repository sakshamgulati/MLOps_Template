import logging

# Now you can import your module
from src.ModelOps.model_fit import ModelFit
from src.DataOps.feature_engg import feature_engg_class

logging.info("Internal modules loaded")

 
stock_data = feature_engg_class()
data=stock_data.request_stock_price_hist('AAPL')
stock_data.data=data
train,test=stock_data.split(use_prophet=True)
logging.info("Data split successfully")
model=ModelFit()
#save reference data for monitoring
#note that reference data is to be replaced while model retraining 
model.save_reference_data(train)
model.data_quality_check(train)
#make predictions
fitted_model,forecast=model.model(train, test)
model.save_model_to_registry(fitted_model)