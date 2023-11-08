import sys
import logging

sys.path.insert(0, "C:/Users/saksh/MLOps_Template")
from src.ModelOps import ModelFit
from src.DataOps import feature_engg_class

logging.info("Internal modules loaded")


diabetes = feature_engg_class()
data, target = diabetes.load_data()
data = diabetes.standard_scaling(data)
X_train, X_test, y_train, y_test = diabetes.split(data, target)
logging.info(f"length of the train set:{len(X_train)}")

# random forest regressor with default parameters
rf_reg = ModelFit()
model = rf_reg.model(X_train, X_test, y_train, y_test)
rf_reg.save_model(model)
logging.info("Model Saved")
