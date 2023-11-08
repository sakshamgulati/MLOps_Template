import pandas as pd
import numpy as np
import logging
import confuse
from sklearn import preprocessing, datasets
from sklearn.model_selection import train_test_split
import wandb
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

logging.basicConfig(level=logging.INFO)


class feature_engg_class:
    """
    #description for this class
    This class is used to load the data,
    scale the data and split the data into train and test set.
    ################################
    #Parameters
    config_file: str
        Path to the config file
    ################################
    #Returns
    None
    """

    def __init__(self, config_file="./conf/smartprice.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = confuse.Configuration("SmartPrice", __name__)
        self.config.set_file(config_file)
        self.DRIVER_PATH = self.config["DRIVER_PATH"].get(str)
        self.diabetes = datasets.load_diabetes()

    def load_data(self):
        """
        #description for this function
        This function is used to load the data from the sklearn datasets.
        ################################
        #Parameters
        None
        ################################
        #Returns
        data: numpy array
            The data from the sklearn datasets
        target: numpy array
        """
        data = self.diabetes.data
        target = self.diabetes.target
        logging.info("Data loaded from sklearn datasets")
        imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
        data_imputed = imputer.fit_transform(data)
        logging.info("Data imputed for missing values")
        return data_imputed, target

    def standard_scaling(self, data):
        """
        #description for this function
        This function is used to scale the data using StandardScaler.
        ################################
        #Parameters
        data: numpy array
            The data from the sklearn datasets
        ################################
        #Returns
        data_scaled: numpy array
            The scaled data

        """
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        logging.info("Data Scaled")
        return data_scaled

    def one_hot_encode(self, data):
        """
        #description for this function
        This function is used to one hot encode the data.
        ################################
        #Parameters
        data: numpy array
            The data from the sklearn datasets
        ################################
        #Returns
        data_encoded: numpy array
            The one hot encoded data

        """
        encoder = OneHotEncoder(sparse=False)
        data_encoded = encoder.fit_transform(data)
        return data_encoded

    def split(self, data, target):
        """
        #description for this function
        This function is used to split the data into train and test set.
        ################################
        #Parameters
        data: numpy array
            The data from the sklearn datasets
        target: numpy array
        ###############################
        #Returns
        X_train: numpy array
            The training data
        X_test: numpy array
            The testing data
        y_train: numpy array
            The training target
        y_test: numpy array
            The testing target
        """
        X_train, X_test, y_train, y_test = train_test_split(
            data, target, test_size=0.33, random_state=42
        )
        logging.info("Data split into testing and train")
        return X_train, X_test, y_train, y_test
