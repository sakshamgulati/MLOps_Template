import pandas as pd
import logging
from sklearn import preprocessing, datasets
from sklearn.model_selection import train_test_split
import wandb
import requests
from tqdm import tqdm
logging.basicConfig(level=logging.INFO)
import os
from dotenv import load_dotenv


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
        self.data = None
        logging.info("Feature Engineering class initialized")

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
        if self.data is None:
            logging.error("Data not loaded")
            #raise exception, throw error
            return None
        self.data = self.data.reset_index()
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

        
    def request_stock_price_hist(self,symbol):
        load_dotenv()
        q_string = 'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={}&outputsize=full&apikey={}'
        
        logging.info("Retrieving stock price data from Alpha Vantage (This may take a while)...")
        ALPHA_VANTAGE_API_KEY=os.getenv("ALPHA_VANTAGE_API_KEY")
        r = requests.get(q_string.format(symbol, ALPHA_VANTAGE_API_KEY))
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
        logging.info("Data loaded successfully")
        logging.info("Data shape: {}".format(df.shape))
        logging.info(f"Earliest Date:{df.index[-1]},Latest Date:{df.index[-1]}")
        return df