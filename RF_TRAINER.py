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
import seaborn as sns
from vwap import *
from utils import *
import pickle

##NOTE: REsults are wrong, need to know actual bid and ask sizes at bar closes.


# PRE-CONFIGURE PANDAS
# Outputs all coloumns of the dataframe in the console
# Expands console char display limit

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

# LOAD DATA

df = pd.read_csv('AAPL.USUSD_Candlestick_5_M_ASK_26.01.2017-11.05.2019.csv', sep=',')
df2 = pd.read_csv('AAPL.USUSD_Candlestick_5_M_BID_26.01.2017-11.05.2019.csv', sep=',')


# REVERSES DATA ORDER IN DF AND KEEPS STARTING INDEX AT 0

#df.iloc[:] = df.iloc[::-1].values
#df2.iloc[:] = df2.iloc[::-1].values

# CLEANS DATAFRAMES

df = df.replace(['-'], np.nan)
df2 = df2.replace(['-'], np.nan)

df = df.dropna()
df2 = df2.dropna()


# FEATURE CONSTRUCTION

df['Close_ask'] = df['Close']
df['Open_ask'] = df['Open']
df['High_ask'] = df['High']
df['Low_ask'] = df['Low']
df['Volume_ask'] = df['Volume']

df['Close_bid'] = df2['Close']
df['Open_bid'] = df2['Open']
df['High_bid'] = df2['High']
df['Low_bid'] = df2['Low']
df['Volume_bid'] =df2['Volume']

df['Close_ask_change'] = df['Close_ask'].pct_change()
df['Close_ask_accel'] = df['Close_ask_change'].pct_change()
df['Close_bid_change'] = df['Close_bid'].pct_change()
df['Close_bid_accel'] = df['Close_bid_change'].pct_change()
df['Spread'] = abs(df['Close_ask'] - df['Close_bid'])
df['Spread_change'] = df['Spread'].pct_change()
df['Total_vol'] = df['Volume_ask'] + df['Volume_bid']
df['Total_vol_change'] = df['Total_vol'].pct_change()
df['Total_vol_accel'] = df['Total_vol_change'].pct_change()

#df['vwap'] =vwap(df['Close'], df['High'], df['Low'], df['Total_vol'], period = 5)
#df['vwap'] = df['vwap'].pct_change().round(5)

df['High_wick_p'] = np.where(df['Close'] > df['Open'], (df['High'] - df['Close']) / (df['High'] - df['Low']), (df['High'] - df['Open']) / (df['High'] - df['Low'])).round(5)
df['Low_wick_p'] = np.where(df['Close'] > df['Open'], (df['Open'] - df['Low']) / (df['High'] - df['Low']), (df['Close'] - df['Low']) / (df['High'] - df['Low'])).round(5)
df['Body_p'] = 1 - df['High_wick_p'] - df['Low_wick_p']

df['RSI_perc_change'] = rsi(df['Close'], n=14).pct_change().round(5)
df['ATR_perc_change'] = average_true_range(df['High'], df['Low'], df['Close'], n=5).pct_change().round(5)
df['SMA_10_perc_change'] = ema_indicator(df['Close'], n=10).pct_change().round(5)
df['SMA_5_perc_change'] = ema_indicator(df['Close'], n=5).pct_change().round(5)

df['RSI_accel'] = df['RSI_perc_change'].pct_change().round(5)
df['ATR_accel'] = df['ATR_perc_change'].pct_change().round(5)
df['SMA_10_accel'] = df['SMA_10_perc_change'].pct_change().round(5)
df['SMA_5_accel'] = df['SMA_5_perc_change'].pct_change().round(5)




# DATA OPTION IF REQUIRED
# print(pd.DatetimeIndex(df['Date']).month)

# CLEAN DATA AGAIN BEFORE PASSING TO MACHINE LEARNING MODEL

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

# DECLARE X VARIABLE FOR TRAINING MODEL
# TRAINING MODEL ON FEATURES CREATED ABOVE

X = df[['Close_bid_change',
        'Close_ask_change',
        'Total_vol_change',

        'High_wick_p',
        'Low_wick_p',
        'Body_p',

        'RSI_perc_change',
        'ATR_perc_change',
        'SMA_10_perc_change',
        'SMA_5_perc_change'

        ]]

# DECLARE Y VARIABLE. ACTUAL LABELS OF THE DATA. TRAINS MODEL TO LOOK FOR THESE SPECIFIED INSTANCES.
# Last value is Nan due to no future value to use to compare the previous close to so take the Nan and set it
# to 0, don't drop it because then the length of the X data (instances) won't match the y data (labels), 0 causes no
# effect on final PnL

#y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

df.loc[df['Close_bid'].shift(-1) > df['Close_ask']+0.05, 'Labels' ] = 1
df.loc[df['Close_ask'].shift(-1) < df['Close_bid']-0.05, 'Labels'] = -1
# df.loc[df['Close_ask'].shift(-1) == df['Close_bid'], 'Labels'] = 0
# df.loc[df['Close_bid'].shift(-1) == df['Close_ask'], 'Labels'] = 0
df['Labels'] = df['Labels'].replace([np.nan], 0)

y = df['Labels']

# FEATURE IMPORTANCE GRAPH

feat = ExtraTreesClassifier()
model = feat.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

# CORRELATION MATRIX GRAPH

corr = df.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)


# SCALE DATA BEFORE FEEDING IT TO THE MODEL.

standardscaler = StandardScaler()
X = standardscaler.fit_transform(X)

# DECLARE TRAIN TEST SPLITS. SHUFFLE = FALSE TO MAKE SURE DATA IS IN CHRONOLOGICAL ORDER

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, shuffle=False)
print(X_train)

# CHECKS FOR NULL VALUES IN THE DATASET BEFORE PASSING TO MODEL
#print(pd.isnull(X_train).sum() > 0)

# TRAINS MODEL

model = RandomForestClassifier(n_estimators=500, n_jobs=-1, min_samples_leaf=10)
model = model.fit(X_train,y_train)

# PRINTS MODEL METRICS

accuracy = metrics.accuracy_score(y_test,model.predict(X_test))
print (accuracy)

report = metrics.classification_report(y_test, model.predict(X_test))
print (report)

confusion_matrix = metrics.confusion_matrix(y_test, model.predict(X_test))
print(confusion_matrix)

# RECORDS LENGTH F TRAINING DATA SO IT KNOWS THE INDEX WHERE THE TEST DATA STARTS. MAKES SURE TO ONLY CALCULATE
# RETURNS FROM THE TEST DATA ONWARDS. THIS IS WHY SHUFFLE = FALSE IN THE TRAIN-TEST SPLIT PARAMETERS
length_x = len(X)
length_x_train = len(X_train)


# STORES PREDICTIONS IN DATAFRAME

df['Predicted'] = model.predict(X)

# CALCULATES UNDERLYING SECURITY PERFORMANCE BY TAKING A CUM. SUM OF THE VALUES IN THE DATAFRAME

cum_underlying_returns = np.cumsum(df[length_x_train:]['Close_ask_change'])

# CALCULATES STRATEGY RETURNS BY TAKING A CUM.SUM OF THE RETURNS.
# NEGATIVE UNDERLYING RETURNS * NEGATIVE (SELL) PREDICTION) = POSITIVE RETURN
# POSITIVE UNDERLYING RETURNS * POSITIVE (BUY) PREDICTION) = POSITIVE RETURN
# NEGATIVE UNDERLYING RETURNS * POSITIVE (BUY) PREDICTION) = NEGATIVE RETURN
# POSITIVE UNDERLYING RETURNS * NEGATIVE (SELL) PREDICTION) = NEGATIVE RETURN


SLIPPAGE = 0.00

df.loc[df['Close_bid'] > df['Close_ask'].shift(1), 'Strat_Returns'] = ((df['Close_bid']-SLIPPAGE - df['Close_ask'].shift(1)) / (df['Close_ask'].shift(1))) * df['Predicted'].shift(1)
df.loc[df['Close_ask'] < df['Close_bid'].shift(1), 'Strat_Returns'] = ((df['Close_ask']+SLIPPAGE - df['Close_bid'].shift(1)) / (df['Close_bid'].shift(1))) * df['Predicted'].shift(1)
df.loc[df['Close_bid'] < df['Close_ask'].shift(1), 'Strat_Returns'] = ((df['Close_bid']-SLIPPAGE - df['Close_ask'].shift(1)) / (df['Close_ask'].shift(1))) * df['Predicted'].shift(1)
df.loc[df['Close_ask'] > df['Close_bid'].shift(1), 'Strat_Returns'] = ((df['Close_ask']+SLIPPAGE - df['Close_bid'].shift(1)) / (df['Close_bid'].shift(1))) * df['Predicted'].shift(1)

df.loc[df['Close_bid'] == df['Close_ask'].shift(1), 'Strat_Returns'] = 0 #((df['Close_bid']-SLIPPAGE - df['Close_ask'].shift(1)) / (df['Close_ask'].shift(1)))  # Even accounts for slippage when trying to sell it back to break even, ensures a loss, used = 0 before but this doesn't account for a micro loss
df.loc[df['Close_ask'] == df['Close_bid'].shift(1), 'Strat_Returns'] = 0 #((df['Close_ask']+SLIPPAGE - df['Close_bid'].shift(1)) / (df['Close_bid'].shift(1))) # Even accounts for slippage when trying to buy it back to break even, ensures a loss, used = 0 before but this account account for a micro loss


cum_strat_returns = np.cumsum(df[length_x_train:]['Strat_Returns'])

print(df.tail(20))

# PLOTS RESULTS
plt.figure(figsize=(10,5))
plt.plot(cum_underlying_returns, color='r',label = 'Underlying Returns')
plt.plot(cum_strat_returns, color='g', label = 'Strategy Returns')
plt.legend()
plt.show()



#Save scaler object
with open('RF_standardscaler_file.p', 'wb') as RF_standardscaler_file, open('RF_model_file.p', 'wb') as RF_model:
    pickle.dump(standardscaler, RF_standardscaler_file)
    pickle.dump(model, RF_model)







