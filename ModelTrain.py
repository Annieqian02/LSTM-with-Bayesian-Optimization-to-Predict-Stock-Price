from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import load_model
from datetime import datetime
import pdb
import glob
import os
import os.path

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

# Building the RNN
from keras.models import Sequential
from keras.layers import Dense, Bidirectional, LSTM
from keras import regularizers
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Building the CNN
import keras
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers import Flatten

import DataRequest

class ProgressCallback(keras.callbacks.Callback):
    def __init__(self, progress, **kwargs):
        super().__init__(**kwargs)
        self.epochs = 0
        self.steps = 0
        self.total_steps = 0
        self.current_step = 0
        self.progress = progress
        self.current_epoch = -1

    def set_params(self, params):
        print("set_params:", params)
        self.verbose = params["verbose"]
        self.epochs = params["epochs"]
        if "steps" in params:
            self.steps = params["steps"]
            self.total_steps = self.epochs * self.steps

    def on_train_begin(self, batch, logs=None):
        # print("train starts")
        self.current_step = 0
        self.progress(0, self.total_steps)
    
    def on_train_end(self, batch, logs=None):
        # print("train ends")
        # self.current_step = self.epochs * self.steps
        self.progress(self.epochs, self.epochs)
    
    def on_epoch_begin(self, batch, logs=None):
        # print("epoch begins")
        self.current_epoch += 1
        self.progress(self.current_epoch, self.epochs)

    def on_train_batch_begin(self, batch, logs=None):
        # self.progress(batch, self.epochs)
        pass
    
    def on_train_batch_end(self, batch, logs=None):
        self.current_step += 1
        # self.progress(self.current_step, self.total_steps)

class ModelTrain():
    def __init__(self, stock, data_size):
        self.stock = stock
        self.data_size = data_size
        self.df = DataRequest.DataHist(self.stock, self.data_size, 'Standard Indicators').RequestFinaldf()
        self.data = self.df.to_numpy()
        self.epochs = 50  # epochs
        self.time_steps = 30  # time_steps
        self.indicator = 'Standard Indicators'

    def SetTimeSteps(self, timestep):
        self.time_steps = timestep

    def SetAlgorithm(self, algorithm):
        self.algorithm = algorithm

    def Setepochs(self, epochs):
        self.epochs = epochs

    def SetIndicator(self, indicator):
        self.indicator = indicator

    def SetDataSize(self, data_size):
        self.data_size = data_size
        self.df = DataRequest.DataHist(self.stock, self.data_size, self.indicator).RequestFinaldf()

    def ComputeValue(self):
        if self.indicator == 'Standard Indicators':
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'RA_5', 'RA_10', 'MACD', 'CCI_20', 'ATR', 'BollingerUpper', 'BollingerLower',
                             'MA_5', 'MA_10', 'Momentum_30', 'Momentum_90', 'ROC', 'WPR_14']
        elif self.indicator == 'Standard Indicators with Stochastic Oscillator':
            self.features = ['Open', 'High', 'Low', 'Close', 'Volume',
                             'RA_5', 'RA_10', 'MACD', 'CCI_20', 'ATR', 'BollingerUpper', 'BollingerLower',
                             'MA_5', 'MA_10', 'Momentum_30', 'Momentum_90', 'ROC', 'WPR_14', 'SOD_14']

        self.num_of_features = len(self.features)
        self.X = self.df[self.features].values
        self.y = self.df['Close'].values
        self.num_of_sample = len(self.X)
        print("Number of Sample: ", self.num_of_sample)

        ####### Data Split into : Train, Test, Validation ########
        self.X_new = self.X
        self.y_new = self.y
        self.num_train = int(round(8 * self.num_of_sample / 10, 0))
        self.num_dev = int(round(self.num_train + self.num_of_sample / 10, 0))

        # print(num_train, num_dev)
        self.raw_X_train = self.X_new[: self.num_train, ]
        self.raw_X_dev = self.X_new[self.num_train:self.num_dev, ]
        self.raw_X_test = self.X_new[self.num_dev:, ]

        self.raw_y_train = self.y_new[:self.num_train, ]
        self.raw_y_dev = self.y_new[self.num_train:self.num_dev, ]
        self.raw_y_test = self.y_new[self.num_dev:, ]

        print("X y train: ", self.raw_X_train.shape, self.raw_y_train.shape)
        print("X y dev: ", self.raw_X_dev.shape, self.raw_y_dev.shape)
        print("X y test: ", self.raw_X_test.shape, self.raw_y_test.shape)

        ####### Data Scaler #######
        self.X_sc = MinMaxScaler(feature_range=(0, 1))
        self.X_MinMax = self.X_sc.fit_transform(self.raw_X_train)

        self.y_sc = MinMaxScaler(feature_range=(0, 1))
        self.y_MinMax = self.y_sc.fit_transform(np.array(self.raw_y_train).reshape(-1, 1))

        self.X_MinMax_dev = self.X_sc.transform(self.raw_X_dev)
        self.y_MinMax_dev = self.y_sc.transform(np.array(self.raw_y_dev).reshape(-1, 1))

        self.X_MinMax_test = self.X_sc.transform(self.raw_X_test)
        self.y_MinMax_test = self.y_sc.transform(np.array(self.raw_y_test).reshape(-1, 1))

        print("X y train MinMaxScaler: ", self.X_MinMax.shape, self.y_MinMax.shape)
        print("X y dev MinMaxScaler: ", self.X_MinMax_dev.shape, self.y_MinMax_dev.shape)
        print("X y test MinMaxScaler: ", self.X_MinMax_test.shape, self.y_MinMax_test.shape)

        self.X_train = []
        self.y_train = []

        for i in range(self.time_steps, len(self.raw_X_train)):
            self.X_train.append(self.X_MinMax[i - self.time_steps:i])
            self.y_train.append(self.y_MinMax[i])

        self.X_train, self.y_train = np.array(self.X_train), np.array(self.y_train)

        self.X_dev = []
        self.y_dev = []

        for i in range(self.time_steps, len(self.raw_X_dev)):
            self.X_dev.append(self.X_MinMax_dev[i - self.time_steps:i])
            self.y_dev.append(self.y_MinMax_dev[i])

        self.X_dev, self.y_dev = np.array(self.X_dev), np.array(self.y_dev)

        print("X y train After time_steps: ", self.X_train.shape, self.y_train.shape)
        print("X y dev After time_steps: ", self.X_dev.shape, self.y_dev.shape)

    def DeleteFile(self):
        # delete all the model in the output file before add new module
        filelist = glob.glob('./output/*')
        for f in filelist:
            os.remove(f)
        print('Deleted')

    def TrainModel(self, progress):
        def build_regressor():
            if self.algorithm == "CNN":

                # since predicting a continuous value, dealing with continuous values
                regressor = Sequential()

                regressor.add(Conv1D(filters=128, kernel_size=2, activation='relu',
                                     input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
                regressor.add(Conv1D(filters=128, kernel_size=2, activation='relu'))

                regressor.add(MaxPooling1D(pool_size=2))

                regressor.add(Flatten())

                regressor.add(Dense(100, activation='relu'))

                regressor.add(Dense(units=1))

                return regressor
            elif self.algorithm == "LSTM":

                # since predicting a continuous value, dealing with continuous values
                regressor = Sequential()

                # adding first LSTM and bias to avoid overfitting (units = num_neurons)
                regressor.add(LSTM(units=200, return_sequences=True,
                                   input_shape=(self.X_train.shape[1], self.X_train.shape[2]),
                                   bias_regularizer=regularizers.l2(1e-6)))

                # second LSTM layer
                regressor.add(LSTM(units=200,
                                   return_sequences=True,
                                   bias_regularizer=regularizers.l2(1e-6)))

                # third LSTM layer
                regressor.add(LSTM(units=200,
                                   return_sequences=True,
                                   bias_regularizer=regularizers.l2(1e-6)))

                # fourth LSTM layer
                regressor.add(LSTM(units=200,
                                   return_sequences=True,
                                   bias_regularizer=regularizers.l2(1e-6)))

                # fifth LSTM layer
                regressor.add(LSTM(units=200,
                                   bias_regularizer=regularizers.l2(1e-6)))

                # adding the output layer
                regressor.add(Dense(units=1))

                return regressor

            elif self.algorithm == "BiLSTM":

                num_layer = 2

                num_neuron = 200

                regressor = Sequential()

                regressor.add(Bidirectional(LSTM(units=num_neuron, return_sequences=True),
                                            input_shape=(self.X_train.shape[1], self.X_train.shape[2])))

                if num_layer > 2:
                    for i in range(0, num_layer - 2):
                        regressor.add(Bidirectional(LSTM(units=num_neuron, return_sequences=True)))

                regressor.add(Bidirectional(LSTM(units=num_neuron)))

                regressor.add(Dense(units=1))

                return regressor

        pg = ProgressCallback(progress)

        # save one model
        callbacks = [ModelCheckpoint(filepath='./output/weights.{epoch:02d}.hdf5', save_best_only=True,
                                     save_weights_only=False, monitor='val_loss', mode='min'), pg]
        # compiling the regressor
        regressor = build_regressor()

        # compiling the regressor, optimizer adam, for regression, loss fxn = mse
        regressor.compile(optimizer='adam', loss='mean_squared_error')

        # summary
        # regressor.summary()

        # Fitting RNN
        self.history = regressor.fit(self.X_train, self.y_train, epochs=self.epochs,
                                     validation_data=(self.X_dev, self.y_dev), callbacks=callbacks)

    def PlotModel(self):
        # summarize history for accuracy
        plt.ylim(0, 0.03)
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model accuracy')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['loss', 'vall_loss'], loc='upper right')

    def GetModel(self, progress):
        # log to file
        # with open('out.log', 'w') as f:
        #     with redirect_stdout(f):
        self.ComputeValue()
        self.DeleteFile()
        self.TrainModel(progress)
        # self.PlotModel()
        list_of_files = glob.glob('./output/*')  # * means all if need specific format then *.csv
        latest_file = str(max(list_of_files, key=os.path.getctime))
        print(os.path.basename(latest_file))
        return latest_file, self.history.history['loss'], self.history.history['val_loss']

    # Load a exist model file
    def GetOriModel(self):
        self.ComputeValue()
        list_of_files = glob.glob('./output/*')  # * means all if need specific format then *.csv
        latest_file = str(max(list_of_files, key=os.path.getctime))
        print(os.path.basename(latest_file))
        return latest_file

    def ModelTest(self, modelfile):
        self.X_test = []
        self.y_test = []

        self.test_dates = self.df['date'].values[self.num_dev:]
        # keep only date 
        self.t = self.test_dates.astype('datetime64[D]')
        self.date_test = []

        for i in range(self.time_steps, len(self.raw_y_test)):
            self.X_test.append(self.X_MinMax_test[i - self.time_steps:i])
            self.y_test.append(self.y_MinMax_test[i])
            self.date_test.append(self.t[i])

        self.X_test, self.y_test = np.array(self.X_test), np.array(self.y_test)
        self.date_test = np.array(self.date_test)
        print(self.X_test.shape)
        print(self.y_test.shape)
        print(self.date_test.shape)
        print()

        """load data and get test result"""
        model = load_model(modelfile)
        # The LSTM needs data with the format of [samples, time steps and features]
        self.y_pred = model.predict(self.X_test)

        self.y_test_in = self.y_sc.inverse_transform(self.y_test)
        self.y_pred_in = self.y_sc.inverse_transform(self.y_pred)
        # viewing results

        self.fig, self.ax = plt.subplots(figsize=(15, 7))
        self.ax.plot(self.date_test, self.y_test_in, color='red', label=f'Real {self.stock} Stock Price')
        self.ax.plot(self.date_test, self.y_pred_in, color='blue', label=f'Predicted {self.stock} Stock Price')
        plt.title(f'{self.stock} Stock Price Prediction')
        plt.xlabel('Time')
        plt.ylabel(f'{self.stock} Stock Price')
        plt.legend()
        # plt.savefig('./output/stock_price_plot.png')
        score = np.sqrt(metrics.mean_squared_error(self.y_pred, self.y_test))
        print('RMSE after {} epochs = '.format(self.epochs), score)
        print('MSE after {} epochs = '.format(self.epochs), score * score)


'''
if __name__ == "__main__":
    MT = ModelTrain('aapl')
    # MT.SetAlgorithm('LSTM')
    modelfile = MT.GetModel()
    MT.ModelTest(modelfile)
'''
