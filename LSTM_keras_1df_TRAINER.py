import numpy as np
import pandas as pd
import math
import sklearn
import sklearn.preprocessing
import datetime
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import model_from_json
import pickle
import sys

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
np.set_printoptions(threshold=sys.maxsize)

# split data in 80%/10%/10% train/validation/test sets
valid_set_size_percentage = 20
test_set_size_percentage = 20

# epoch 1 batch 10 , 5min, neurons, 400, 200, 100, 50, the one currently being tested

# LOAD DATA
df = pd.read_csv('AAPL.USUSD_Candlestick_5_M_ASK_15.03.2017-30.05.2019.csv')
df2 = pd.read_csv('AAPL.USUSD_Candlestick_5_M_BID_15.03.2017-30.05.2019.csv')


# Merges dataframes based on if they have the same dates so gets rid of values where date/time isn;t in the other one,
# making the dataframes of equal length and merged into one. labels change to open_x and open_y, etc... automatically.
# Also sets the index to the Local time coloumn in the dataframe

df = df2.merge(df, how = 'inner', on = ['Local time'])
df = df.set_index('Local time')

# Gets total volume and discards the individual volumes for bid and ask

df['Total_vol'] = (df['Volume_x'] + df['Volume_y']).pct_change().round(5)
#df = df.drop(columns=['Volume_x', 'Volume_y'])

# ONLY TAKES DATA WE WANT TO TRAIN ON
df = df[[
    'Open_x',
    'High_x',
    'Low_x',
    'Close_x',
    'Total_vol'
       ]]


# CLEANS DATAFRAMES
df = df.dropna()

# SCALERS TO MAKE VALUES BETWEEN 0 ADN 1
price_scaler = sklearn.preprocessing.MinMaxScaler()
total_vol_scaler = sklearn.preprocessing.MinMaxScaler()

#open_x refers to the open of data in the df2 which is the BID data

df['Open_x'] = price_scaler.fit_transform(df['Open_x'].values.reshape(-1,1))
df['High_x'] = price_scaler.fit_transform(df['High_x'].values.reshape(-1,1))
df['Low_x'] = price_scaler.fit_transform(df['Low_x'].values.reshape(-1,1))
df['Close_x'] = price_scaler.fit_transform(df['Close_x'].values.reshape(-1, 1))
df['Total_vol'] = total_vol_scaler.fit_transform(df['Total_vol'].values.reshape(-1, 1))


# mode 0 means returns prediction for all features in transformed form
# mode 1 makes prediction based on 0 and 1

def load_data(df, seq_len, mode = 0):
    data_raw = df.values  # convert to numpy array
    data = []

    # create all possible sequences of length seq_len
    for index in range(len(data_raw) - seq_len):
        data.append(data_raw[index: index + seq_len])

    data = np.array(data)
    valid_set_size = int(np.round(valid_set_size_percentage / 100 * data.shape[0]))
    test_set_size = int(np.round(test_set_size_percentage / 100 * data.shape[0]))
    train_set_size = data.shape[0] - (valid_set_size + test_set_size)

    # :train_set_size represents the number of batches used, :-1 represents the nmber of values in each batch used so
    # all the values except the last one in the seq_len, : represents all the coloumns

    x_train = data[:train_set_size, :-1, :]
    y_train = data[:train_set_size, -1, :]

    x_valid = data[train_set_size:train_set_size + valid_set_size, :-1, :]
    y_valid = data[train_set_size:train_set_size + valid_set_size, -1, :]

    x_test = data[train_set_size + valid_set_size:, :-1, :]
    y_test = data[train_set_size + valid_set_size:, -1, :]

    if mode == 0:

        return [x_train, y_train, x_valid, y_valid, x_test, y_test]

    elif mode == 1:

        # '3' represents the closing price feature

        y_train_int = []
        for i in range(x_train.shape[0]):
            if y_train[i, 3] > x_train[i, -1, 3]:
                y_train_int.append(1)
            elif y_train[i, 3] < x_train[i, -1, 3]:
                y_train_int.append(0)
            else:
                y_train_int.append(2)
        y_train = np.array(y_train_int).reshape(-1, 1)

        y_valid_int = []
        for i in range(x_valid.shape[0]):
            if y_valid[i, 3] > x_valid[i, -1, 3]:
                y_valid_int.append(1)
            elif y_valid[i, 3] < x_valid[i, -1, 3]:
                y_valid_int.append(0)
            else:
                y_valid_int.append(2)
        y_valid = np.array(y_valid_int).reshape(-1, 1)

        y_test_int = []
        for i in range(x_test.shape[0]):
            if y_test[i, 3] > x_test[i, -1, 3]:
                y_test_int.append(1)
            elif y_test[i, 3] < x_test[i, -1, 3]:
                y_test_int.append(0)
            else:
                y_test_int.append(2)
        y_test = np.array(y_test_int).reshape(-1, 1)

        return [x_train, y_train, x_valid, y_valid, x_test, y_test]


# create train, test data

seq_len = 25 # choose sequence length
num_features = 5
x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(df, seq_len, mode=0)


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint


#print(x_train.shape[0], x_train.shape[1], num_features)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, epsilon=0.0001, patience=3, verbose=1)
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

EPOCHS = 3
BATCH_SIZE = 1

model = Sequential()

model.add(LSTM(units = 400, return_sequences = True, input_shape = (x_train.shape[1], num_features)))
model.add(Dropout(0.2))
model.add(LSTM(units = 200, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 100, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
model.add(Dense(units = num_features))
model.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

model.fit(x_train,
          y_train,
          epochs = EPOCHS,
          batch_size = BATCH_SIZE,
          validation_data=(x_valid, y_valid),
          callbacks = [checkpoint, lr_reduce],
          shuffle= True)


y_train_pred = model.predict(x_train)
y_valid_pred = model.predict(x_valid)
y_test_pred = model.predict(x_test)


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
corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,3]-y_valid[:,0]),
            np.sign(y_valid_pred[:,3]-y_valid_pred[:,0])).astype(int)) / y_valid.shape[0]
corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,3]-y_test[:,0]),
            np.sign(y_test_pred[:,3]-y_test_pred[:,0])).astype(int)) / y_test.shape[0]

# threshold = price_scaler.transform(np.array([0]).reshape(-1,1))
# print(threshold)
#
#
# corr_price_development_train = np.sum(np.equal(np.sign(y_train[:,3]-threshold[0,0]),
#             np.sign(y_train_pred[:,3]-threshold[0,0])).astype(int)) / y_train.shape[0]
# corr_price_development_valid = np.sum(np.equal(np.sign(y_valid[:,3]-threshold[0,0]),
#             np.sign(y_valid_pred[:,3]-threshold[0,0])).astype(int)) / y_valid.shape[0]
# corr_price_development_test = np.sum(np.equal(np.sign(y_test[:,3]-threshold[0,0]),
#             np.sign(y_test_pred[:,3]-threshold[0,0])).astype(int)) / y_test.shape[0]

print('correct sign prediction for close - open price for train/valid/test: %.2f/%.2f/%.2f'%(
    corr_price_development_train, corr_price_development_valid, corr_price_development_test))


# serialize model to JSON
model_json = model.to_json()
with open("LSTM_model.json", "w") as json_file:
    json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("LSTM_model.h5")
print("Saved model to disk")

# Save scaler objects
with open('total_vol_scaler_file.p', 'wb') as total_vol_scaler_file,\
    open('price_scaler_file.p', 'wb') as price_scaler_file:

    pickle.dump(price_scaler, price_scaler_file)
    pickle.dump(total_vol_scaler, total_vol_scaler_file)



