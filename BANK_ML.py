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




# Displays all coloumns in df when outputting but some coloumns may go on the second line due to the console char width
# limit so you also expand the width so this doesn;t happen

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('float_format', '{:f}'.format)

df = pd.read_csv('JPM.USUSD_Candlestick_1_M_ASK_22.02.2017-04.05.2019.csv', sep=',')
df2 = pd.read_csv('WFC.USUSD_Candlestick_1_M_ASK_22.02.2017-04.05.2019.csv', sep=',')


df = df.dropna()
df2 = df2.dropna()


# Features construction, volume perc change should be 1 or -1 so its normalized
df['High_wick_p'] = np.where(df['Close'] > df['Open'], (df['High'] - df['Close']) / (df['High'] - df['Low']), (df['High'] - df['Open']) / (df['High'] - df['Low']))
df['Low_wick_p'] = np.where(df['Close'] > df['Open'], (df['Open'] - df['Low']) / (df['High'] - df['Low']), (df['Close'] - df['Low']) / (df['High'] - df['Low']))
df['Body_p'] = 1 - df['High_wick_p'] - df['Low_wick_p']

df2['High_wick_p'] = np.where(df2['Close'] > df2['Open'], (df2['High'] - df2['Close']) / (df2['High'] - df2['Low']), (df2['High'] - df2['Open']) / (df2['High'] - df2['Low']))
df2['Low_wick_p'] = np.where(df2['Close'] > df2['Open'], (df2['Open'] - df2['Low']) / (df2['High'] - df2['Low']), (df2['Close'] - df2['Low']) / (df2['High'] - df2['Low']))
df2['Body_p'] = 1 - df2['High_wick_p'] - df2['Low_wick_p']

df['High_wick_p2'] = df2['High_wick_p']
df['Low_wick_p2'] = df2['Low_wick_p']
df['Body_p2'] = df2['Body_p']


df['Volume'] = df['Volume']/1000000
df['Volume'] = df['Volume'].round(4)

df['Vol_perc_change'] = df['Volume'].pct_change()
df['RSI_perc_change'] = rsi(df['Close'], n=14).pct_change()
df['ATR_perc_change'] = average_true_range(df['High'], df['Low'], df['Close'], n=5).pct_change()
df['SMA_10_perc_change'] = ema_indicator(df['Close'], n=10).pct_change()
df['SMA_5_perc_change'] = ema_indicator(df['Close'], n=5).pct_change()
df['Stock_Returns'] = df['Close'].pct_change()

df['Volume2'] = df2['Volume']/1000000
df['Volume2'] = df['Volume2'].round(4)

df['Vol_perc_change2'] = df['Volume2'].pct_change()
df['RSI_perc_change2'] = rsi(df2['Close'], n=14).pct_change().round(3)
df['ATR_perc_change2'] = average_true_range(df2['High'], df2['Low'], df2['Close'], n=5).pct_change()
df['SMA_10_perc_change2'] = ema_indicator(df2['Close'], n=10).pct_change()
df['SMA_5_perc_change2'] = ema_indicator(df2['Close'], n=5).pct_change()
df['Stock_Returns2'] = df2['Close'].pct_change()


df['Ratio'] = (df['Close'] / df2['Close']).pct_change()
df['Diff'] = (df['Close'] - df2['Close']).pct_change()

df['vwap1'] = vwap(df['Close'], df['High'], df['Low'], df['Volume'], period = 5)
df['vwap1'] = df['vwap1'].pct_change()

df['vwap2'] = vwap(df2['Close'], df2['High'], df2['Low'], df2['Volume'], period = 5)
df['vwap2'] = df['vwap2'].pct_change()


df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()


X = df[['RSI_perc_change',
        'ATR_perc_change',
        'Stock_Returns',
        'SMA_10_perc_change',
        'SMA_5_perc_change',
        'High_wick_p',
        'Low_wick_p',
        'Body_p',
        'vwap1',


        'RSI_perc_change2',
        'ATR_perc_change2',
        'Stock_Returns2',
        'SMA_10_perc_change2',
        'SMA_5_perc_change2',
        'High_wick_p2',
        'Low_wick_p2',
        'Body_p2',
        'vwap2',


        'Ratio'



        ]]

#y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

# 6 works very well as well
# 5 cents works well, but need to double check the spread is small for jpm, 7 woks vey well as well, 8 wroks well too but 7 works better


df.loc[df['Close'].shift(-1) > df['Close'] + 0.07, 'Predicted' ] = 1
df.loc[df['Close'].shift(-1) < df['Close'] - 0.07, 'Predicted' ] = -1
df.loc[df['Close'].shift(-1) == df['Close'], 'Predicted'] = 0
df['Predicted'] = df['Predicted'].replace([np.nan], 0)
print(df.head(50))

y = df['Predicted']

feat = ExtraTreesClassifier()
model = feat.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

corr = df.corr()
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns)

standardscaler = StandardScaler(with_mean=False, with_std=False)
X = standardscaler.fit_transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.60, shuffle=False)

#print(pd.isnull(X_train).sum() > 0)

log_regr = RandomForestClassifier(n_estimators=500, n_jobs=-1, min_samples_leaf=10)

log_regr = log_regr.fit(X_train,y_train)



accuracy = metrics.accuracy_score(y_test,log_regr.predict(X_test))
print (accuracy)

report = metrics.classification_report(y_test, log_regr.predict(X_test))
print (report)
length = len(X_train)


df['Predicted'] = log_regr.predict(X)
cum_stock_returns = np.cumsum(df[length:]['Stock_Returns'])
df['Strat_returns'] = df['Stock_Returns'] * df['Predicted'].shift(1)
cum_strat_returns = np.cumsum(df[length:]['Strat_returns'])


plt.figure(figsize=(10,5))
plt.plot(cum_stock_returns, color='r',label = 'Stock Returns')
plt.plot(cum_strat_returns, color='g', label = 'Strategy Returns')

plt.legend()
plt.show()

print(df.tail(40))
