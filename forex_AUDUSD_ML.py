import numpy as np
from ta import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn import metrics
from sklearn.neighbors import *
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import *
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC,SVC
import backtrader as bt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics
import datetime
from vwap import *
from utils import *


# PRE-CONFIGURE PANDAS
# Outputs all coloumns of the dataframe in the console
# Expands console char display limit

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# LOAD DATA

df = pd.read_csv('AUDUSD_Candlestick_5_M_ASK_08.06.2016-11.05.2019.csv', sep=',')
df2 = pd.read_csv('XAUUSD_Candlestick_5_M_ASK_08.06.2016-11.05.2019.csv', sep=',')

# REVERSES DATA ORDER IN DF AND KEEPS STARTING INDEX AT 0

#df.iloc[:] = df.iloc[::-1].values
#df2.iloc[:] = df2.iloc[::-1].values

# CLEANS DATAFRAMES

df = df.replace(['-'], np.nan)
df2 = df2.replace(['-'], np.nan)

df = df.dropna()
df2 = df2.dropna()


# FEATURE CONSTRUCTION, FIRST DATASET

SPREAD = 0.0002

df['AUDUSD_perc_change'] = df['Close'].pct_change()
df['Close_spread_adj'] = df['Close'] - SPREAD
df['AUDUSD_perc_change_spread_adj'] = (df['Close_spread_adj'] - df['Close'].shift(1))/df['Close'].shift(1)

df['High_wick_p'] = np.where(df['Close'] > df['Open'], (df['High'] - df['Close']) / (df['High'] - df['Low']), (df['High'] - df['Open']) / (df['High'] - df['Low']))
df['Low_wick_p'] = np.where(df['Close'] > df['Open'], (df['Open'] - df['Low']) / (df['High'] - df['Low']), (df['Close'] - df['Low']) / (df['High'] - df['Low']))
df['Body_p'] = 1 - df['High_wick_p'] - df['Low_wick_p']

df['RSI'] = rsi(df['Close'], n=14).pct_change()

# FEATURE CONSTRUCTION, SECOND DATASET
# df2['Volume'] = df2['Volume'].str.replace('K', '0').astype(float)

df['GOLD_change'] = df2['Close'].pct_change()
df['GOLD_vol_change'] = df2['Volume'].pct_change()

df2['High_wick_p'] = np.where(df2['Close'] > df2['Open'], (df2['High'] - df2['Close']) / (df2['High'] - df2['Low']), (df2['High'] - df2['Open']) / (df2['High'] - df2['Low']))
df2['Low_wick_p'] = np.where(df2['Close'] > df2['Open'], (df2['Open'] - df2['Low']) / (df2['High'] - df2['Low']), (df2['Close'] - df2['Low']) / (df2['High'] - df2['Low']))
df2['Body_p'] = 1 - df2['High_wick_p'] - df2['Low_wick_p']
df['High_wick_p2'] = df2['High_wick_p']
df['Low_wick_p2'] = df2['Low_wick_p']
df['Body_p2'] = df2['Body_p']

df['GOLD_RSI'] = rsi(df2['Close'], n=14).pct_change()
df['GOLD_vwap'] = vwap(df2['Close'], df2['High'], df2['Low'], df2['Volume'], period = 5)
df['GOLD_vwap'] = df['GOLD_vwap'].pct_change().round(3)


# DATA OPTION IF REQUIRED
# print(pd.DatetimeIndex(df['Date']).month)


# CLEAN DATA AGAIN BEFORE PASSING TO MACHINE LEARNING MODEL

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# DECLARE X VARIABLE FOR TRAINING MODEL
# TRAINING MODEL ON FEATURES CREATED ABOVE

X = df[[
        'High_wick_p',
        'Body_p',
        'Low_wick_p',
        'RSI',
        'AUDUSD_perc_change',

        'High_wick_p2',
        'Low_wick_p2',
        'Body_p2',
        'GOLD_change',
        'GOLD_vol_change',
        'GOLD_RSI',
        'GOLD_vwap'

        ]]


# DECLARE Y VARIABLE. ACTUAL LABELS OF THE DATA. TRAINS MODEL TO LOOK FOR THESE SPECIFIED INSTANCES.

#y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

df.loc[df['Close_spread_adj'].shift(-1) > df['Close'] + 0.0002 , 'Labels' ] = 1
df.loc[df['Close_spread_adj'].shift(-1) < df['Close'] - 0.0002 , 'Labels'] = -1
df.loc[df['Close_spread_adj'].shift(-1) == df['Close'], 'Labels'] = 0
df['Labels'] = df['Labels'].replace([np.nan], 0)

y = df['Labels']

# FEATURE IMPORTANCE GRAPH

feat = ExtraTreesClassifier()
model = feat.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# SCALE DATA BEFORE FEEDING IT TO THE MODEL.

standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)

# DECLARE TRAIN TEST SPLITS. SHUFFLE = FALSE TO MAKE SURE DATA IS IN CHRONOLOGICAL ORDER

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

# CHECKS FOR NULL VALUES IN THE DATASET BEFORE PASSING TO MODEL
#print(pd.isnull(X_train).sum() > 0)

# TRAINS MODEL

MODEL = RandomForestClassifier(n_estimators=500, n_jobs=-1, min_samples_leaf=10)
MODEL = MODEL.fit(X_train,y_train)

# PRINTS MODEL METRICS

accuracy = metrics.accuracy_score(y_test,MODEL.predict(X_test))
print (accuracy)

report = metrics.classification_report(y_test, MODEL.predict(X_test))
print (report)

confusion_matrix = metrics.confusion_matrix(y_test, MODEL.predict(X_test))
print(confusion_matrix)

# RECORDS LENGTH F TRAINING DATA SO IT KNOWS THE INDEX WHERE THE TEST DATA STARTS. MAKES SURE TO ONLY CALCULATE
# RETURNS FROM THE TEST DATA ONWARDS. THIS IS WHY SHUFFLE = FALSE IN THE TRAIN-TEST SPLIT PARAMETERS

length = len(X_train)

# STORES PREDICTIONS IN DATAFRAME

df['Predicted'] = MODEL.predict(X)

# CALCULATES UNDERLYING SECURITY PERFORMANCE BY TAKING A CUM. SUM OF THE VALUES IN THE DATAFRAME

cum_underlying_returns = np.cumsum(df[length:]['AUDUSD_perc_change'])

# CALCULATES STRATEGY RETURNS BY TAKING A CUM.SUM OF THE RETURNS.
# NEGATIVE UNDERLYING RETURNS * NEGATIVE (SELL) PREDICTION) = POSITIVE RETURN
# POSITIVE UNDERLYING RETURNS * POSITIVE (BUY) PREDICTION) = POSITIVE RETURN
# NEGATIVE UNDERLYING RETURNS * POSITIVE (BUY) PREDICTION) = NEGATIVE RETURN
# POSITIVE UNDERLYING RETURNS * NEGATIVE (SELL) PREDICTION) = NEGATIVE RETURN

df['Strat_returns'] = df['AUDUSD_perc_change_spread_adj'] * df['Predicted'].shift(1)
cum_strat_returns = np.cumsum(df[length:]['Strat_returns'])


# PLOTS RESULTS
plt.figure(figsize=(10,5))
plt.plot(cum_underlying_returns, color='r',label = 'Underlying Returns')
plt.plot(cum_strat_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()


