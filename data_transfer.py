import pandas as pd
import sys

class Category():
    def __init__(self, dataframe):        
        self.dataframe = dataframe

    '''
    @classmethod
    def read_data(self, file_name):
        dataset = pd.read_csv(file_name)
        return dataset
    '''
    
    def createDataset(self):
        '''        
        main function to execute all other functions       
        
        '''
        self.RA_5()
        self.RA_10()
        self.MACD()          
        self.CCI()        
        self.ATR()
        self.BOLL()
        self.MA()
        self.MTM()
        self.ROC()
        self.WPR()

    def createDataset_SOD(self):
        '''
        main function to execute all other functions

        '''
        self.RA_5()
        self.RA_10()
        self.MACD()
        self.CCI()
        self.ATR()
        self.BOLL()
        self.MA()
        self.MTM()
        self.ROC()
        self.WPR()
        self.SOD()

  
    def RA_5(self):
        '''
        rolling average: stdev in 5 day windows
        
        returns pandas dataframe with added column 'RA'
        '''       
        
        self.dataframe['RA_5'] = self.dataframe['Adj Close'].rolling(window = 5).std()
        
    def RA_10(self):
        '''
        rolling average: stdev in 10 day windows
        
        returns pandas dataframe with added column 'RA'
        '''       
        
        self.dataframe['RA_10'] = self.dataframe['Adj Close'].rolling(window = 10).std()        
        

    def MACD(self):
        '''        
        MACD: (12-day EMA - 26-day EMA)
        EMA stands for Exponential Moving Average.  
        
        http://www.andrewshamlet.net/2017/01/19/python-tutorial-macd-moving-average-convergencedivergence/        
        '''        
        self.dataframe['MACD'] = pd.Series.ewm(self.dataframe['Adj Close'], span=12).mean() - \
                            pd.Series.ewm(self.dataframe['Adj Close'], span=26).mean()
    
    def CCI(self):
        """Calculate Commodity Channel Index for given data.
        
        :param df: pandas.DataFrame
        :param n: 
        :updates pandas.DataFrame
        http://www.andrewshamlet.net/2017/07/08/python-tutorial-cci/       
        """
        
        n = 20
        # Typical Price (TP) = (High + Low + Close)/3
        TP = (self.dataframe['High'] + self.dataframe['Low'] + self.dataframe['Close']) / 3
        
        # CCI = (Typical Price  -  n-period SMA of TP) / (Constant x Mean Deviation)
        CCI = pd.Series((TP - TP.rolling(n, min_periods=n).mean()) / (0.015*TP.rolling(n, min_periods=n).std()),
                        name='CCI_' + str(n))
        self.dataframe = self.dataframe.join(CCI)
    
    def ATR(self):
        """
        
        :param df: pandas.DataFrame
        :param n: 
        :updates pandas.DataFrame
        
        """
        n = 14
        i = 0
        TR_l = [0]
        while i < self.dataframe.index[-1]:
            TR = max(self.dataframe.loc[i + 1, 'High'], self.dataframe.loc[i, 'Close']) - \
                 min(self.dataframe.loc[i + 1, 'Low'], self.dataframe.loc[i, 'Close'])
            TR_l.append(TR)
            i = i + 1
        TR_s = pd.Series(TR_l)
        ATR = pd.Series(TR_s.ewm(span=n, min_periods=n).mean(), name='ATR')
        self.dataframe = self.dataframe.join(ATR)
            
            
    def BOLL(self):
        """
        :param dataframe: pandas.DataFrame
        :param n: 
        :updates pandas.DataFrame
        
        https://towardsdatascience.com/trading-technical-analysis-with-pandas-43e737a17861        
        """
       
        MA = pd.Series(self.dataframe['Close'].rolling(window=20).mean())
        MSD = pd.Series(self.dataframe['Close'].rolling(window=20).std())
        b1 =  MA + 2 * MSD
        B1 = pd.Series(b1, name='BollingerUpper')
        self.dataframe = self.dataframe.join(B1)
        
        b2 = MA - 2 * MSD
        B2 = pd.Series(b2, name='BollingerLower')
        self.dataframe = self.dataframe.join(B2)            
        
    def MA(self):
        """
        Calculate the moving average for the given data.
        
        :param df: pandas.DataFrame
        :param n: 
        """
        n = 5
        MA = pd.Series(self.dataframe['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        self.dataframe = self.dataframe.join(MA)
        
        n = 10
        MA = pd.Series(self.dataframe['Adj Close'].rolling(n, min_periods=n).mean(), name='MA_' + str(n))
        self.dataframe = self.dataframe.join(MA)

    def MTM(self):
        """
        Calculate Momentum
        
        :param dataframe: pandas.DataFrame 
        :param n: 
        :return: pandas.DataFrame
        """
        n = 30
        M = pd.Series(self.dataframe['Adj Close'].diff(n), name='Momentum_' + str(n))
        self.dataframe = self.dataframe.join(M)
        
        n = 90
        M = pd.Series(self.dataframe['Adj Close'].diff(n), name='Momentum_' + str(n))
        self.dataframe = self.dataframe.join(M)        
            
    def ROC(self):
        """
        Calculates Rate of Change
        :param dataframe: pandas.DataFrame
        :param n: 
        https://github.com/Crypto-toolbox/pandas-technical-indicators/blob/master/technical_indicators.py
        
        """
        n = 30
        M = self.dataframe['Adj Close'].diff(n )
        N = self.dataframe['Adj Close'].shift(n )
        ROC = pd.Series(M / N, name='ROC')
        self.dataframe = self.dataframe.join(ROC)
    
    def WPR(self):
        n = 14
        WPR = pd.Series((self.dataframe['Close'] - self.dataframe['Low'].rolling(n).min())/
                        (self.dataframe['High'].rolling(n).max() - self.dataframe['Low'].rolling(n).min())*100, name = "WPR_%s" % str(n))
        self.dataframe = self.dataframe.join(WPR)

    def SOD(self):
        """
        :param df: pandas.DataFrame
        :param n:
        :return: pandas.DataFrame
        """
        n = 14
        SOk = pd.Series((self.dataframe['Close'] - self.dataframe['Low']) / (self.dataframe['High'] - self.dataframe['Low']), name='SO%k')
        SOd = pd.Series(SOk.ewm(span=n, min_periods=n).mean(), name='SOD_' + str(n))
        self.dataframe = self.dataframe.join(SOd)


'''if __name__ == "__main__":
    
    dataframe = sys.argv[1] # csv of raw data downloaded from Yahoo Finance
    output_csv = sys.argv[2]     

    c = Category(dataframe)
    c.createDataset()
    c.df.to_csv(output_csv)
'''  
    
