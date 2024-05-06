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
import requests
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
import os

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

    def __init__(self):
        from dotenv import load_dotenv
        load_dotenv()
        ALPHA_VANTAGE_API_KEY=os.getenv("ALPHA_VANTAGE_API_KEY")
        self.data =self.__request_stock_price_hist('IBM', ALPHA_VANTAGE_API_KEY)
        logging.info("Data loaded successfully")
        logging.info("Data shape: {}".format(self.data.shape))
        logging.info(f"Earliest Date:{self.data.index[-1]},Latest Date:{self.data.index[-1]}")


    def split(self, use_prophet=True):
        """
        #description for this function
        This function is used to split the data into train and test set on 70-30 split.
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
        if use_prophet:
            self.data=self.data[['date','close']]
            self.data.columns = ['ds','y']
            logging.info("Data prepared for Prophet")
        
        train=self.data[:int(0.7*self.data.shape[0])]
        test=self.data[int(0.7*self.data.shape[0]):]
        logging.info("Data split into train and test successfully")
        logging.info(f"Train shape: {train.shape[0]}")
        logging.info(f"Test shape: {test.shape[0]}")
        return train, test
    
    def __request_stock_price_hist(symbol, token, sample = False):
            if sample == False:
                q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={}&outputsize=full&apikey={}'
            else:
                q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={}&apikey={}'

            logging.info("Retrieving stock price data from Alpha Vantage (This may take a while)...")
            r = requests.get(q_string.format(symbol, token))
            logging.info("Data has been successfully downloaded...")
            date = []
            colnames = list(range(0, 6))
            df = pd.DataFrame(columns = colnames)
            logging.info("Storing the retrieved data into a dataframe...")
            for i in tqdm(r.json()['Weekly Adjusted Time Series'].keys()):
                date.append(i)
                row = pd.DataFrame.from_dict(r.json()['Weekly Adjusted Time Series'][i], orient='index').reset_index().T[1:]
                df = pd.concat([df, row], ignore_index=True)
            df.columns = ["open", "high", "low", "close", "adjusted close", "volume", "dividend amount"]
            df['date'] = date
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            df = df.sort_index(ascending=True)
            #change the data types of the columns
            df = df.apply(pd.to_numeric, errors='coerce')
            logging.info("Sorting the index and changing column data types...")

            return df
