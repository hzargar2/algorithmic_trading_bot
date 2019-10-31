from pandas import read_csv
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn import metrics
import pandas as pd
import numpy as np
from ta import *


# Function that calls ARIMA model to fit and forecast the data
def StartARIMAForecasting(Actual, P, D, Q):
    model = ARIMA(Actual, order=(P, D, Q))
    model_fit = model.fit(disp=0)
    prediction = model_fit.forecast()[0]
    return prediction


# Get exchange rates
ActualData = pd.read_csv('COCOA.CMDUSD_Candlestick_5_M_ASK_04.12.2017-03.05.2019.csv')
ActualData['Returns'] = ActualData['Close'].pct_change()
ActualData['Returns'] = ActualData['Returns'].round(5)

ActualData = ActualData.replace([np.inf, -np.inf], np.nan)
ActualData = ActualData.dropna()

ActualData = ActualData['Returns'].values

np.seterr (all='ignore')

# Size of exchange rates
NumberOfElements = len(ActualData)

# Use 70% of data as training, rest 30% to Test model
TrainingSize = int(NumberOfElements * 0.7)
TrainingData = ActualData[0:TrainingSize]
TestData = ActualData[TrainingSize:NumberOfElements]

# new arrays to store actual and predictions
Actual = [x for x in TrainingData]
Predictions = list()

# in a for loop, predict values using ARIMA model
for timepoint in range(len(TestData)):
    try:
        ActualValue = TestData[timepoint]
        # forcast value
        Prediction = StartARIMAForecasting(Actual, 3, 1, 0)
        print('Actual=%f, Predicted=%f' % (ActualValue, Prediction))
        # add it in the list
        Predictions.append(Prediction)
        Actual.append(ActualValue)
    except:
        pass


df = pd.DataFrame()
df['Predictions'] = pd.Series(Predictions)
df['Actual'] = pd.Series(Actual)
#print(df)
df['Predicted (1 or -1)'] = np.where(df['Predictions'] > df['Actual'].shift(1), 1, -1)
df['Actual (1 r -1)'] = np.where(df['Actual'] > df['Actual'].shift(1), 1, -1)
# shift Predictions up one when comparing agaisnt other classifier so you can comare the against teh same return shift(-1) later
df['True'] = np.where (df['Actual (1 r -1)'] == df['Predicted (1 or -1)'], 1, 0)

sum = np.sum(df['True'])
prob_right = sum/len(df['True'])

print(sum)
print(prob_right * 100)

# Print MSE to see how good the model is

if len(Actual)>len(Predictions):
    err = metrics.mean_squared_error(Actual[:len(Predictions)], Predictions)
else:
    err = metrics. mean_squared_error(Actual, Predictions[:len(Actual)])
print(err)

# plot
pyplot.plot(TestData)
pyplot.plot(Predictions, color='red')
pyplot.show()