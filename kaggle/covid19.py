import numpy as np
import pandas as pd
import os
import warnings
from plotly.graph_objs import Figure, Scatter

from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
class Covid19ARIMA():
    def __init__(self):
        self._csv = pd.read_csv(os.path.join(os.curdir, 'kaggle', 'covid19', 'input', 'time_series_covid_19_confirmed.csv'), na_values=['Unknown', '?', 'na', 'nan', 'NaN'])
        self._csv['Province/State'] = self._csv['Province/State'].fillna('Unknown')
        self._csv.drop(labels=['Lat', 'Long'], axis=1, inplace=True)
        self._csv.set_index(['Province/State', 'Country/Region'], inplace=True, drop=True)
        self.fitted = False
    def select(self, province, country):
        self._select = self._csv.loc[(province, country),]
        self._select.index = pd.to_datetime(self._select.index)
        # self._select.reset_index(drop=True, inplace=True)
        self._select = self._select.diff().fillna(self._select[0])
        self._select = self._select/1000
        return self
    def predict(self, start=None, end=None, testing=False):
        if self.fitted:
            if testing:
                interval = self._select.index[-self.test_size:]
                start = interval[0]
                end = interval[-1]
                history = self._select[-self.test_size:]
            forecast = self.res.predict(start=start, end=end)
            if testing:
                loss = np.sqrt(mean_squared_error(history, forecast))
                print(f'Scor model (RMSE): {loss}')
                return [history, forecast]
            return forecast
    def fit(self, test_size=28, ar=1, ma=1):
        if not self.fitted:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model = ARIMA(self._select[:-test_size], order=(ar,1,ma), freq='D')
                self.res = self.model.fit()
            self.fitted = True
            self.test_size=test_size
            return self.res
        else: return None
    def plot(self, serieses=[]):
        splot = Figure()
        for i, series in enumerate(serieses):
            splot.add_trace(Scatter(x=series.index, y=series.values, name=f'Cazuri confirmate {i+1}'))
        splot.update_layout(xaxis_title='Numar zile trecute', yaxis_title='Numar cazuri la 1000 loc', title='Cazuri confirmate covid19')
        splot.show()
    def get_selectie(self):
        return self._select

import silence_tensorflow.auto
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
class Covid19ANN():
    def __init__(self):
        self._csv = pd.read_csv(os.path.join(os.curdir, 'kaggle', 'covid19', 'input', 'time_series_covid_19_confirmed.csv'), na_values=['Unknown', '?', 'na', 'nan', 'NaN'])
        self._csv['Province/State'] = self._csv['Province/State'].fillna('Unknown')
        self._csv.drop(labels=['Lat', 'Long'], axis=1, inplace=True)
        self._csv.set_index(['Province/State', 'Country/Region'], inplace=True, drop=True)
        self.fitted = False
        self.been_build = False
        self.window = None
        self.predictions = None
    def select(self, province, country):
        self._select = self._csv.loc[(province, country),]
        self._select.reset_index(drop=True)
        self._select = self._select.diff().fillna(self._select[0])
        self._select = self._select/1000
        self._select.index = range(0, len(self._select))
        return self
    def plot(self, serieses=[]):
        splot = Figure()
        for i, series in enumerate(serieses):
            splot.add_trace(Scatter(x=series.index, y=series.values, name=f'Cazuri confirmate {i+1}'))
        splot.update_layout(xaxis_title='Numar zile trecute', yaxis_title='Numar cazuri la 1000 loc', title='Cazuri confirmate covid19')
        splot.show()
    def make_train_test_data(self, window=1, predictions=140):
        self.window, self.predictions = (window, predictions)
        x = self._select.to_numpy()
        self._min, self._max = np.min(x), np.max(x)
        x = (x-self._min)/(self._max-self._min)
        X, Y = [], []
        for entry in range(0, len(x)-window, 1):
            X.append([list(x[entry:entry+window])])
            Y.append([x[entry+window]])
        X, Y = np.array(X), np.array(Y)
        print(X.shape, Y.shape)
        return train_test_split(X, Y, shuffle=False, test_size=predictions)
    def build(self, window=1, verbose=1, predictions=140, dropout=0.2, layers=[8]):
        self.window, self.predictions = (window, predictions)
        self.been_build = True
        self.model = Sequential(name='ModelPredictie')
        self.model.add(Dense(layers[0], activation='relu', input_dim=self.window))
        self.model.add(Dropout(dropout))
        for layer in layers[1:]:
            self.model.add(Dense(layer, activation='relu'))
            self.model.add(Dropout(dropout))
        self.model.add(Dense(1, name='output'))
        self.model.compile(optimizer='adam', loss='mse')
        if verbose: self.model.summary()
    def train(self, epochs=15, verbose=1):
        X_train, X_test, Y_train, Y_test = self.make_train_test_data(window=self.window, predictions=self.predictions)
        if not self.been_build: self.build()
        self.model.fit(x=X_train, y=Y_train, validation_data=(X_test, Y_test), verbose=verbose, epochs=epochs, shuffle=True)
        self.fitted = True
        self.X_test, self.Y_test = X_test, Y_test
    def test(self, window=1, predictions=140):
        if self.fitted == True:
            loss = np.sqrt(self.model.evaluate(self.X_test, self.Y_test, verbose=0))
            print('---------------------------------------------------------------')
            print(f'Scor pe setul de testare (RMSE): {loss}')
            print('Testare cu fereastra deplasabila')
            print('---------------------------------------------------------------')
            test = []
            for sample in self.X_test[:window]:
                test.append(sample[0][-1])
            test = pd.Series(np.array(test).flatten(), dtype=np.float32)
            y_pred = []
            for d in range(predictions):
                pred = self.model.predict([list(test)]).flatten()
                y_pred.append(pred)
                test = test[1:].append(pd.Series(pred, dtype=np.float32))
            y_pred = np.array(y_pred).flatten()
            Y_test = self.Y_test.flatten()
            loss = np.sqrt(mean_squared_error(Y_test, y_pred))
            print(f'Scor pe setul de testare (RMSE): {loss}')
            y_redresat = self.model.predict(self.X_test).flatten()
            return pd.Series(y_redresat*(self._max-self._min)+self._min, dtype=np.float32), pd.Series(y_pred*(self._max-self._min)+self._min, dtype=np.float32), pd.Series(Y_test*(self._max-self._min)+self._min, dtype=np.float32)
    def get_selectie(self):
        return self._select