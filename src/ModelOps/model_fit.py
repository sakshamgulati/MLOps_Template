import logging
import confuse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
class model_fit():
    def __init__(self,config_file="./conf/smartprice.yaml"):
        self.logger=logging.getLogger(__name__)
        self.config = confuse.Configuration("Smartprice", __name__)
        self.config.set_file(config_file)
        # self.DRIVER_PATH = self.config["DRIVER_PATH"].get(str)

    def model(self,X_train,X_test,y_train,y_test):
        reg = LinearRegression().fit(X_train, y_train)
        print(reg.score(X_train, y_train))
        y_pred = reg.predict(X_test)
        assert len(y_pred) == len(y_test), "Length mismatch"
        print("Mean squared error: %.2f" % mean_absolute_error(y_test, y_pred))
        return mean_absolute_error(y_test, y_pred)