import pandas as pd
import numpy as np
import logging
import confuse
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import wandb


class feature_engg_class:
    def __init__(self, config_file="./conf/smartprice.yaml"):
        self.logger = logging.getLogger(__name__)
        self.config = confuse.Configuration("SmartPrice", __name__)
        self.config.set_file(config_file)
        self.DRIVER_PATH = self.config["DRIVER_PATH"].get(str)

    def load_data(self):
        df = pd.read_csv(self.DRIVER_PATH, on_bad_lines="skip", usecols=range(16))
        wandb.init(project="ml-ops-template")

        cols_to_remove = [
            "vin",
            "seller",
            "year",
            "saledate",
            "make",
            "model",
            "trim",
            "state",
        ]
        df.drop(cols_to_remove, axis=1, inplace=True)
        df["condition"] = df["condition"].astype("category")
        cat_col = [col for col in df.columns if df[col].dtype == "object"]
        cat_df = df[cat_col]
        num_df = df.drop(cat_col, axis=1)
        output = num_df["sellingprice"]
        num_df = num_df.drop(["sellingprice"], axis=1)
        num_df = num_df.drop(["condition"], axis=1)
        wandb.log({"Training Size": df.shape[0], "numerical Size": num_df.shape[1]})
        return num_df, cat_df, output

    def min_max_scaling(self, cat_df, num_df):
        one_hot_df = pd.get_dummies(cat_df).reset_index()
        x = num_df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        num_df = pd.DataFrame(x_scaled, columns=num_df.columns)
        fin_df = pd.concat([num_df, one_hot_df], axis=1)
        return fin_df

    def split(self, fin_df, output):
        X_train, X_test, y_train, y_test = train_test_split(
            fin_df, output, test_size=0.33, random_state=42
        )
        return X_train, X_test, y_train, y_test
