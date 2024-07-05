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
    #change the path to the location of the dataset
    df = pd.read_csv(os.path.join("Stocks","a.us.txt"), delimiter=",", usecols=['Date','Open','High','Low','Close'])
    print("Loaded from kaggle repo")

# Sort DataFrame by date
df = df.sort_values('Date')

# Double check the result
df.head()

plt.figure(figsize = (18,9))
plt.plot(range(df.shape[0]),(df['Low']+df['High'])/2.0)
plt.xticks(range(0,df.shape[0],500),df['Date'].loc[::500],rotation=45)
plt.xlabel('Date',fontsize=18)
plt.ylabel('Mid Price',fontsize=18)
plt.show()

high_prices = df.loc[:,"High"].as_matrix()
low_prices = df.loc[:,"Low"].as_matrix()
mid_prices = (high_prices+low_prices)/2.0

train_data = mid_prices[:11000]
test_data = mid_prices[11000:]

#normalise data so that it falls between 0 and 1
scaler = MinMaxScaler()
train_data = train_data.reshape(-1,1)
test_data = test_data.reshape(-1,1)

smoothing_window_size = 2500
for di in range(0,10000,smoothing_window_size):
    scaler.fit(train_data[di:di+smoothing_window_size,:])
    train_data[di:di+smoothing_window_size,:] = scaler.transform(train_data[di:di+smoothing_window_size,:])

scaler.fit(train_data[di+smoothing_window_size:,:])
train_data[di+smoothing_window_size:,:] = scaler.transform(train_data[di+smoothing_window_size:,:])

train_data = train_data.reshape(-1)
test_data = scaler.transform(test_data).reshape(-1)

EMA = 0.0
gamma = 0.1
for ti in range(11000):
    EMA = gamma*train_data[ti] + (1-gamma)*EMA
    train_data[ti] = EMA

all_mid_data = np.concatenate([train_data, test_data], axis=0)

window_size = 100
N = train_data.size
std_avg_predictions = []
std_avg_x = []
mse_errors = []

for pred_idx in range(window_size,N):
    if pred_idx >= N:
        date = dt.datetime.strptime(k, "%Y-%m-%d").date() + dt.timedelta(days=1)
    else:
        date = df.loc[pred_idx, "Date"]