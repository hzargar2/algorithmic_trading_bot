import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 20
test_set_size_percentage = 20

# LOAD DATA
df2 = pd.read_csv('AAPL.USUSD_Candlestick_5_M_ASK_02.11.2017-11.05.2019.csv')
df = pd.read_csv('TXN.USUSD_Candlestick_5_M_ASK_02.11.2017-11.05.2019.csv')

#df = df.drop(['Volume'], axis = 1)
#df2 = df2.drop(['Volume'], axis = 1)
#df['Volume'] = df['Volume'] + df2['Volume']


df['Open'] = df['Open'].pct_change()
df['Close'] = df['Close'].pct_change()
df['High'] = df['High'].pct_change()
df['Low'] = df['Low'].pct_change()
df['Volume'] = df['Volume'].pct_change()

df2['Open'] = df2['Open'].pct_change()
df2['Close'] = df2['Close'].pct_change()
df2['High'] = df2['High'].pct_change()
df2['Low'] = df2['Low'].pct_change()
df2['Volume'] = df2['Volume'].pct_change()

# CLEANS DATAFRAMES
df = df.replace(['-'], np.nan)
df2 = df2.replace(['-'], np.nan)
df = df.dropna()
df2 = df2.dropna()


# Merges dataframes based on if they have the same dates so gets rid of values where date/time isn;t in the other one,
# making the dataframes of equal length and merged into one. labels change to open_x and open_y, etc... automatically.
# Also sets the index to the Local time coloumn in the dataframe

df = df2.merge(df, how = 'inner', on = ['Local time'])
df = df.set_index('Local time')
print(df.head(20))




# each coloumn needs its own seperate minmax scaler objectso that when inverse transform is
# done, it can get back the original value, since volume is last coloumn min max is used, when inverse transform is
# done, it thinks prices are volume so returns the prices as volumes when inverse transform is done, can see the
# diff when the close coloumn is at the end instead. So need a for loop to egnerate each coloumns own minmax scaler

open_x_scaler = sklearn.preprocessing.MinMaxScaler()
high_x_scaler = sklearn.preprocessing.MinMaxScaler()
low_x_scaler = sklearn.preprocessing.MinMaxScaler()
close_x_scaler = sklearn.preprocessing.MinMaxScaler()
volume_x_scaler = sklearn.preprocessing.MinMaxScaler()

open_y_scaler = sklearn.preprocessing.MinMaxScaler()
high_y_scaler = sklearn.preprocessing.MinMaxScaler()
low_y_scaler = sklearn.preprocessing.MinMaxScaler()
close_y_scaler = sklearn.preprocessing.MinMaxScaler()
volume_y_scaler = sklearn.preprocessing.MinMaxScaler()


df['Open_x'] = open_x_scaler.fit_transform(df['Open_x'].values.reshape(-1,1))
df['High_x'] = high_x_scaler.fit_transform(df['High_x'].values.reshape(-1,1))
df['Low_x'] = low_x_scaler.fit_transform(df['Low_x'].values.reshape(-1,1))
df['Close_x'] = close_x_scaler.fit_transform(df['Close_x'].values.reshape(-1, 1))
df['Volume_x'] = volume_x_scaler.fit_transform(df['Volume_x'].values.reshape(-1,1))

df['Open_y'] = open_y_scaler.fit_transform(df['Open_y'].values.reshape(-1,1))
df['High_y'] = high_y_scaler.fit_transform(df['High_y'].values.reshape(-1,1))
df['Low_y'] = low_y_scaler.fit_transform(df['Low_y'].values.reshape(-1,1))
df['Close_y'] = close_y_scaler.fit_transform(df['Close_y'].values.reshape(-1, 1))
df['Volume_y'] = volume_y_scaler.fit_transform(df['Volume_y'].values.reshape(-1,1))


def load_data(stock, seq_len):
    data_raw = stock.as_matrix()  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    return [x_train, y_train, x_valid, y_valid, x_test, y_test]

# normalize stock


# create train, test data
seq_len = 20 # choose sequence length
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df, seq_len)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()

regressor.add(LSTM(units = 200, return_sequences = True, input_shape = (x_train.shape[1], 10)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 25))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 10))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.fit(x_train, y_train, epochs = 1, batch_size = 50)

y_train_pred = regressor.predict(x_train)
y_valid_pred = regressor.predict(x_valid)
y_test_pred = regressor.predict(x_test)
print(y_train_pred)
print(y_test_pred)



ft = 3 # 0 = open, 1 = high, 2 = low, 3 = close, 4 = volume

## show predictions
plt.figure(figsize=(15, 5));
plt.subplot(1,2,1);

plt.plot(np.arange(y_train.shape[0]), y_train[:,ft], color='blue', label='train target')

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_valid.shape[0]), y_valid[:,ft],
         color='gray', label='valid target')

plt.plot(np.arange(y_train.shape[0]+y_valid.shape[0],
                   y_train.shape[0]+y_test.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0]),y_train_pred[:,ft], color='red',
         label='train prediction')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_valid_pred.shape[0]),
         y_valid_pred[:,ft], color='orange', label='valid prediction')

plt.plot(np.arange(y_train_pred.shape[0]+y_valid_pred.shape[0],
                   y_train_pred.shape[0]+y_valid_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('past and future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best');

plt.subplot(1,2,2);

plt.plot(np.arange(y_train.shape[0], y_train.shape[0]+y_test.shape[0]),
         y_test[:,ft], color='black', label='test target')

plt.plot(np.arange(y_train_pred.shape[0], y_train_pred.shape[0]+y_test_pred.shape[0]),
         y_test_pred[:,ft], color='green', label='test prediction')

plt.title('future stock prices')
plt.xlabel('time [days]')
plt.ylabel('normalized price')
plt.legend(loc='best')
plt.show()

corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,3]-y_train[:,0]),
            np.sign(y_train_pred[:,3]-y_train_pred[:,0])).astype(int)) / y_train.shape[0]
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,1]-y_valid[:,0]),
            np.sign(y_valid_pred[:,3]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,1]-y_test[:,0]),
            np.sign(y_test_pred[:,3]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))
