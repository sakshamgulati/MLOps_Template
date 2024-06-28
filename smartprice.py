import logging

# Now you can import your module
from src.ModelOps.model_fit import ModelFit
from src import ModelOps
from src.DataOps.feature_engg import feature_engg_class
import pandas as pd
logging.info("Internal modules loaded")

 
stock_data = feature_engg_class()
data=stock_data.request_stock_price_hist('AAPL')
stock_data.data=data.tail(5)
print(stock_data.data)
stock_data.data = stock_data.data.reset_index()
stock_data.data=stock_data.data[['date','close']]
stock_data.data.columns = ['ds','y']
stock_data.data['ds'] = pd.to_datetime(stock_data.data['ds'])
stock_data.data['y'] = stock_data.data['y'].astype(float)
print(stock_data.data.info())
preds=ModelOps.ModelInference().inference(stock_data.data)
preds=preds[['ds','yhat']]
print("now looking the preds")
preds['yhat']=preds['yhat'].astype(float)
preds['ds']=pd.to_datetime(preds['ds'])
print(preds.shape)
print(preds.head())
