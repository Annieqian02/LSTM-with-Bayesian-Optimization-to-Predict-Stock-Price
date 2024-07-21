# import json
import requests

import numpy as np
import pandas as pd

## alpha vantage
## import alpha_vantage
## from alpha_vantage.timeseries import TimeSeries

## yahoo finance
import yfinance as yf

from data_transfer import *

class DataHist:

    def __init__(self, symbol, data_size, indicator):
        self.symbol = symbol
        self.data_size = data_size
        self.indicator = indicator

    def GetRawData(self):
        df = yf.download(str(self.symbol)).reset_index().iloc[::-1]
        df = df.rename({'Date': 'date'}, axis=1)
        df['Volume'] = df['Volume'].astype(float)
        df = df.round(3)
        df = df.iloc[::-1]
        df = df.reset_index()
        df = df.drop("index", axis=1)
        return df

    ' use data_transfer.py document to compute the necessary data from raw data and output a new dataframe '
    ' Input: a dataframe n*7 '
    ' Output: a dataframe n*20 '

    def DataTransfer(self, df, indicator):
        ####### use data_transfer to transfer the raw data to ready to use data as x*20 or x*21 format #######
        c = Category(df)
        if indicator == 'Standard Indicators':
            c.createDataset()
        elif indicator == 'Standard Indicators with Stochastic Oscillator':
            c.createDataset_SOD()


        ####### since we compute some value, Nan shows in first 90 rows, so delete first 90 rows #######
        c_data = (c.dataframe.iloc[90:, ])

        ####### Check if there is any invalid data in dataframe #######
        if (c_data.isnull().any().any() == True):
            print("There are Nan value in dataset!")
        else:
            return c_data

    def RequestFinaldf(self):
        df = self.DataTransfer(self.GetRawData(), self.indicator)
        if df.shape[0] >= self.data_size:
            df = df[-self.data_size:]

        return df




if __name__ == "__main__":
    c = DataHist('AAPL', 30, 'Standard Indicators')
    # c.SetDataSource("Yahoo Finance") # change data source here
    print(c.RequestFinaldf())

