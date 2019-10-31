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


# Displays all coloumns in df when outputting but some coloumns may go on the second line due to the console char width
# limit so you also expand the width so this doesn;t happen

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)

df = pd.read_csv('USDCAD.csv', sep=',')
df2 = pd.read_csv('US_3M_BOND.csv', sep=',')
df3 = pd.read_csv('CAD_3M_BOND.csv', sep=',')
df4 = pd.read_csv('US_10Y.csv', sep=',')
df5 = pd.read_csv('CAD_10Y.csv', sep=',')
df6 = pd.read_csv('BRENT_OIL_FUTURES.csv', sep=',')
df7 = pd.read_csv('GOLD_FUTURES.csv', sep=',')

# Reverses and preserves the index (keeps starting index at 0)
df.iloc[:] = df.iloc[::-1].values
df2.iloc[:] = df2.iloc[::-1].values
df3.iloc[:] = df3.iloc[::-1].values
df4.iloc[:] = df4.iloc[::-1].values
df5.iloc[:] = df5.iloc[::-1].values
df6.iloc[:] = df6.iloc[::-1].values
df7.iloc[:] = df7.iloc[::-1].values

df6 = df6.replace(['-'], np.nan)
df7 = df7.replace(['-'], np.nan)

df = df.dropna()
df2 = df2.dropna()
df3 = df3.dropna()
df4 = df4.dropna()
df5 = df5.dropna()
df6 = df6.dropna()
df7 = df7.dropna()



# Features construction, volume perc change should be 1 or -1 so its normalized
df['High_wick_p'] = np.where(df['Close'] > df['Open'], (df['High'] - df['Close']) / (df['High'] - df['Low']), (df['High'] - df['Open']) / (df['High'] - df['Low']))
df['Low_wick_p'] = np.where(df['Close'] > df['Open'], (df['Open'] - df['Low']) / (df['High'] - df['Low']), (df['Close'] - df['Low']) / (df['High'] - df['Low']))
df['Body_p'] = 1 - df['High_wick_p'] - df['Low_wick_p']

df['USDCAD_perc_change'] = df['Close'].pct_change()
df['RSI'] = rsi(df['Close'], n=14).pct_change()
df['ATR'] = average_true_range(df['High'], df['Low'], df['Close'], n=10)

df['usd_3m_perc_change'] = df2['Close'].pct_change()
df['cad_3m_perc_change'] = df3['Close'].pct_change()
df['perc_diff_us3m_cad3m'] = df['usd_3m_perc_change'] - df['cad_3m_perc_change']
df['diff_us3m_cad3m'] = df2['Close'] - df3['Close']

df['us10y_perc_change'] = df4['Close'].pct_change()
df['cad10y_perc_change'] = df5['Close'].pct_change()
df['us10_perc_diff_cad10'] = df['us10y_perc_change'] - df['cad10y_perc_change']
df['diff_us10y_cad10y'] = df4['Close'] - df5['Close']

df['us10y_us3m_diff'] = df4['Close'] - df2['Close']
df['us10y_us3m_diff_perc'] = df['us10y_perc_change'] - df['usd_3m_perc_change']
df['cad10y_us3m_diff'] = df5['Close'] - df2['Close']
df['cad10y_us3m_diff_perc'] = df['cad10y_perc_change'] - df['usd_3m_perc_change']
df['us10y_cad3m_diff'] = df4['Close'] - df3['Close']
df['us10y_cad3m_diff_perc'] = df['us10y_perc_change'] - df['usd_3m_perc_change']
df['cad10y_cad3m_diff'] = df5['Close'] - df3['Close']
df['cad10y_cad3m_diff_perc'] = df['cad10y_perc_change'] - df['cad_3m_perc_change']


df6['Volume'] = df6['Volume'].str.replace('K', '0')
df['BRENT_change'] = df6['Close'].pct_change()
df['BRENT_vol_change'] = df6['Volume'].astype(float).pct_change()
df['BRENT_RSI'] = rsi(df6['Close'].astype(float), n=14).pct_change()

df7['Volume'] = df7['Volume'].str.replace('K', '0')
df7['Close'] = df7['Close'].str.replace(',','')
df['GOLD_change'] = df7['Close'].astype(float).pct_change()
df['GOLD_vol_change'] = df7['Volume'].astype(float).pct_change()
df['GOLD_RSI'] = rsi(df7['Close'].astype(float), n=14).pct_change()


print(pd.DatetimeIndex(df['Date']).month)

df = df.replace([np.inf, -np.inf], np.nan)
df = df.dropna()

print(df.head())


X = df[['USDCAD_perc_change',

        'High_wick_p',
        'Body_p',
        'Low_wick_p',
        'RSI',
        'ATR',

        'BRENT_vol_change',
        'BRENT_change',
        'BRENT_RSI',


        ]]

y = np.where(df['Close'].shift(-1) > df['Close'], 1, -1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, shuffle=False)

#print(pd.isnull(X_train).sum() > 0)

log_regr = RandomForestClassifier(n_estimators=5000, n_jobs=-1, min_samples_leaf=10)
log_regr = log_regr.fit(X_train,y_train)



feat = ExtraTreesClassifier()
model = feat.fit(X,y)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind='barh')
plt.show()

accuracy = metrics.accuracy_score(y_test,log_regr.predict(X_test))
print (accuracy)

report = metrics.classification_report(y_test, log_regr.predict(X_test))
print (report)
length = len(X_train)


df['Predicted'] = log_regr.predict(X)
cum_stock_returns = np.cumsum(df[length:]['USDCAD_perc_change'])
df['Strat_returns'] = df['USDCAD_perc_change'] * df['Predicted'].shift(1)
cum_strat_returns = np.cumsum(df[length:]['Strat_returns'])


plt.figure(figsize=(10,5))
plt.plot(cum_stock_returns, color='r',label = 'Stock Returns')
plt.plot(cum_strat_returns, color='g', label = 'Strategy Returns')

plt.legend()
plt.show()

print(df.tail(20))
