import datetime as dt
import json
import os
import urllib.request

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler

data_source = "kaggle"

if data_source == "alphavantage":

    api_key = open("api_key.txt", "r").read()

    # American Airlines stock market prices
    ticker = "AAL"

    # JSON file with all the stock market data for AAL from the last 20 years
    url_string = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=%s&outputsize=full&apikey=%s"%(ticker,api_key)
    
    # Save data to file_to_save
    file_to_save = "stock_market_data-%s.csv"%ticker

    # If data is not already saved
    # Grab data from URl
    # Store values to a Pandas DataFrame
    if not os.path.exists(file_to_save):
        with urllib.request.urlopen(url_string) as url:
            data = json.loads(url.read().decode())
            # Extract stock market data
            data = data["Time Series (Daily)"]
            df = pd.DataFrame(columns=["Date","Low","High","Close","Open"])
            for k,v in data.items():
                date = dt.datetime.strptime(k, "%Y-%m-%d")
                data_row = [date.date(), float(v["3. low"]), float(v["2. high"]),
                            float(v["4. close"]), float(v["1. open"])]
                df.loc[-1:] = data_row
                df.index = df.index + 1
            print("Data saved to : %s"%file_to_save)
            df.to_csv(file_to_save)
    #If data already saved, load from csv
    else:
        print("File exists. Loading data from CSV")
        df = pd.read_csv(file_to_save)

else:   #loading from kaggle
    df = pd.read_csv(os.path.join("Stocks","hpq.us.txt"), delimiter=",", usecols=['Date','Open','High','Low','Close'])
    print("Loaded from kaggle repo")

