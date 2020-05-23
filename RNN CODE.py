# -*- coding: utf-8 -*-
"""
Created on Sat May 16 19:12:38 2020

@author: JOGESH MISHRA
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

training_data= pd.read_csv('Google_Stock_Price_Train.csv')

train_set=training_data.iloc[:,1:2].values

from sklearn.preprocessing import MinMaxScaler
sc= MinMaxScaler(feature_range=(0,1))

train_set=sc.fit_transform(train_set)

X_train=[]
y_train=[]

for i in range(60,1258) :
    X_train.append(train_set[i-60:i,0])
    y_train.append(train_set[i])
    
X_train,y_train = np.array(X_train),np.array(y_train)

X_train= np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))

regressor= Sequential()

regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50,return_sequences=False))
regressor.add(Dropout(0.2))

regressor.add((Dense(units=1)))

regressor.compile(optimizer='adam',loss='mean_squared_error')

regressor.fit(X_train,y_train,batch_size=32,epochs=100)


test_data = pd.read_csv('Google_Stock_Price_Test.csv')

test_set= test_data.iloc[:,1:2].values
test_set.shape

dataset_total = pd.concat((training_data['Open'],test_data['Open']),axis=0)

inputs = dataset_total[len(dataset_total)-len(test_data)-60:].values
inputs=inputs.reshape(-1,1)

inputs= sc.transform(inputs)

X_test=[]

for i in range(60,80):
    X_test.append(inputs[i-60:i,0])

X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

predicted = regressor.predict(X_test)

predicted_stock_price=  sc.inverse_transform(predicted)

plt.plot(test_set,color='red',label='Actual Stock Price')
plt.plot(predicted_stock_price,color='blue',label='Predcited Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

import math

from sklearn.metrics import mean_squared_error

mse=mean_squared_error(test_set,predicted_stock_price)

rmse = math.sqrt(mse)

def build_regressor(optimizer):
    regressor= Sequential()
    regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50,return_sequences=True))
    regressor.add(Dropout(0.2))
    regressor.add(LSTM(units=50,return_sequences=False))
    regressor.add(Dropout(0.2))
    regressor.add((Dense(units=1)))

    regressor.compile(optimizer=optimizer,loss='mean_squared_error')
    return regressor

from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV

regressor= KerasRegressor(build_fn=build_regressor)

parameters={'batch_size':[32,64],
            'epochs':[100,400],
            'optimizer':['adam','rmsprop']
            }
grid_search = GridSearchCV(estimator=regressor,param_grid=parameters,scoring='neg_mean_squared_error',cv=2)

grid_search.fit(X_train,y_train)

grid_search.best_params_
grid_search.best_score_
