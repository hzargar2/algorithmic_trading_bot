import backtrader as bt
import argparse
import pandas
import numpy as np
from datetime import datetime
import pickle
from keras.models import model_from_json
import  pandas as pd
import datetime
import time
import pytz

pd.set_option('display.width', 1000)
np.set_printoptions(suppress=True,linewidth=1000,threshold=1000)

class ML(bt.Strategy):

    params = (
        ('EMERGENCY_STOP',0.01),
    )

    def log(self, txt, dt=None):

        # Logging function for this strategy
        dt = dt or self.datas[0].datetime.datetime(0)
        print('%s, %s' % (dt, txt))

    def __init__(self):

        # LOAD INDICATORS

        self.rsi = bt.indicators.RSI(self.datas[0].close, period=14)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=5)
        self.sma10 = bt.indicators.EMA(self.datas[0].close, period=10)
        self.sma5 = bt.indicators.EMA(self.datas[0].close, period=5)

        # LOADS RF MODEL

        try:

            self.RF_model = pickle.load(open('RF_model_file.p', 'rb'))
            print("RF MODEL SUCCESSFULLY LOADED")

        except OSError as e:
            print('OS Error: {}'.format(e))
            print('RF MODEL FAILED TO LOAD.')

        # LOADS SCALERS FOR RF MODEL

        try:

            self.RF_scaler = pickle.load(open('RF_standardscaler_file.p', 'rb'))
            print("SCALERS SUCCESSFULLY LOADED")

        except OSError as e:
            print('OS Error: {}'.format(e))
            print('SCALERS FAILED TO LOAD.')


        # Dataseries 0 contains 'BID' data from IB
        self.dataclose0 = self.datas[0].close
        self.datahigh0 = self.datas[0].high
        self.dataopen0 = self.datas[0].open
        self.datalow0 = self.datas[0].low
        self.datavolume0 = self.datas[0].volume

        # Dataseries 1 contains 'ASK' data from IB
        self.dataclose1 = self.datas[1].close
        self.datahigh1 = self.datas[1].high
        self.dataopen1 = self.datas[1].open
        self.datalow1 = self.datas[1].low
        self.datavolume1 = self.datas[1].volume

        # Dataseries 1 contains 'TRADES' data from IB
        self.dataclose2 = self.datas[2].close
        self.datahigh2 = self.datas[2].high
        self.dataopen2 = self.datas[2].open
        self.datalow2 = self.datas[2].low
        self.total_volume = self.datas[2].volume * 1000

        # To keep track of pending orders
        self.order = None
        self.data_live = False

        #Stop and profit targets
        self.EMERGENCY_STOP = self.p.EMERGENCY_STOP

        self.bar_lookback = 19

        # Total vol used instead = 9 , if both bid and ask vol then  = 10
        self.features = 9

        # keep track of orders
        self.order_refs = []

        # ID to identify bar
        self.bar_id = None


    def notify_order(self, order):

        # Buy/Sell order submitted/accepted to/by broker - Nothing to do

        if order.status in [order.Submitted, order.Accepted]:
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED @ {}'.format(order.executed.price))
            elif order.issell():
                self.log('SELL EXECUTED @ {}'.format(order.executed.price))

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order. ORDER REF changs to NONE only if the order status is no longer in Submitted
        # or Accepted

        self.order = None

    def notify_data(self, data, status, *args, **kwargs):

        # NOTIFIES WHETHER DATA IS LIVE OR DELAYED

        print('*' * 5, 'DATA NOTIF:', data._getstatusname(status),
              *args)
        if status == data.LIVE:
            self.data_live = True

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    def next(self):

        # SIMPLY LOG PRICES. PRICES ARE LAST TRADE PRICES

        self.log('OPEN: {}, HIGH: {}, LOW: {}, CLOSE: {}, VOLUME: {} '.format(self.dataopen2[0], self.datahigh2[0],
                 self.datalow2[0], self.dataclose2[0], self.total_volume[0]))

#        if len(self.datas[0]) < 15:
#            return

        if (self.datahigh1[0] - self.datalow1[0]) == 0 or\
            self.total_volume[-1] == 0:
            # (abs(self.dataclose0[-1] - self.dataclose1[-1])) == 0:

            return



        Close_bid_change = (self.dataclose0[0]-self.dataclose0[-1])/self.dataclose0[-1]
        round(Close_bid_change,5)


        Close_ask_change = (self.dataclose1[0]-self.dataclose1[-1])/self.dataclose1[-1]
        round(Close_ask_change,5)


        Total_vol_change = (self.total_volume[0]-self.total_volume[-1])/self.total_volume[-1]
        round(Total_vol_change,5)


        # Spread_change = ((abs(self.dataclose0[0] - self.dataclose1[0])) - (abs(self.dataclose0[-1] - self.dataclose1[-1]))) / (abs(self.dataclose0[-1] - self.dataclose1[-1]))
        # round(Spread_change,5)
        #
        # Spread = abs(self.dataclose0[0] - self.dataclose1[0])

        if self.dataclose1[0] > self.dataopen1[0]:

            High_wick_p = ((self.datahigh1[0] - self.dataclose1[0]) / (self.datahigh1[0] - self.datalow1[0]))
            round(High_wick_p,5)

        else:

            High_wick_p = ((self.datahigh1[0] - self.dataopen1[0]) / (self.datahigh1[0] - self.datalow1[0]))
            round(High_wick_p,5)


        if self.dataclose1[0] > self.dataopen1[0]:

            Low_wick_p = ((self.dataopen1[0] - self.datalow1[0]) / (self.datahigh1[0] - self.datalow1[0]))
            round(Low_wick_p,5)


        else:

            Low_wick_p = ((self.dataclose1[0] - self.datalow1[0]) / (self.datahigh1[0] - self.datalow1[0]))
            round(Low_wick_p,5)

        Body_p = 1 - High_wick_p - Low_wick_p
        round(Body_p,5)
        RSI_perc_change = ((self.rsi[0] - self.rsi[-1])/(self.rsi[-1]))
        round(RSI_perc_change,5)
        ATR_perc_change = ((self.atr[0] - self.atr[-1]) / (self.atr[-1]))
        round(ATR_perc_change,5)
        SMA_10_perc_change = ((self.sma10[0] - self.sma10[-1]) / (self.sma10[-1]))
        round(SMA_10_perc_change,5)
        SMA_5_perc_change = ((self.sma5[0] - self.sma5[-1]) / (self.sma5[-1]))
        round(SMA_5_perc_change,5)


        data = {
                'Close_bid_change':[Close_bid_change],
                'Close_ask_change':[Close_ask_change],
                'Total_vol_change':[Total_vol_change],
                # 'Spread_change':[Spread_change],
                # 'Spread':[Spread],
                'High_wick_p':[High_wick_p],
                'Low_wick_p':[Low_wick_p],
                'Body_p':[Body_p],
                'RSI_perc_change':[RSI_perc_change],
                'ATR_perc_change':[ATR_perc_change],
                'SMA_10_perc_change':[SMA_10_perc_change],
                'SMA_5_perc_change':[SMA_5_perc_change]
                }

        X = pd.DataFrame(data)

        # CHECKS TO EE IF DATA IS LIVE BEFORE CONTINUING
        # IMPLEMENT safe fault for disconnects later

        if not self.data_live:
            return

        # CANCELS ANY PENDING ORDERS THAT DID NOT ET EXECUTED BEFORE PLACING ANOTHER ORDER

        if self.order:
            self.cancel(self.order)

        # CLOSES POSITION AS SOON AS THE NEXT BAR IS RECEIVED

        if self.position:
            if self.position.size > 0:
                self.log('SELL CREATE @ {}'.format(self.dataclose0[0])) # SELL AT BID
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell(size=self.position.size, transmit=True, exectype=bt.Order.Market)
                #self.order_refs.append(self.order)

            elif self.position.size < 0:
                self.log('BUY CREATE @ {}'.format(self.dataclose1[0])) # BUY AT ASK
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy(size=self.position.size, transmit=True, exectype=bt.Order.Market)
                #self.order_refs.append(self.order)

        # BEFORE PLACING TRADES< IT CHECKS TO SEE IF WE ARE IN THE PROPER TRADING HOURS
        # BEFORE 9:30AM, IT WAITS FOR MARKET TO OPEN
        # AFTER 3:45PM, STOPS SCRIPT AND TRADING STOPS AND END OF TRADING DAY APPROACHES

        if self.datas[0].datetime.time() < datetime.time(9, 30):
            # don't operate until the market opens
            return

        # 15:45 = 3:45pm
        elif self.datas[0].datetime.time() >= datetime.time(15, 45):
            print('END OF TRADING DAY APPROACHING. TRADING STOPPED AT ', self.data.datetime.time())
            exit()

        # FEEDS MODELS
        X = self.RF_scaler.transform(X)
        pred = self.RF_model.predict(X)
        print(pred)

        time.sleep(2)

        # Check if we are in the market and if an order already been executed on the same bar
        # len() keeps track of the number of bars that have been processed so far

        if not self.position and self.bar_id != len(self.datas[0]):

            if pred == 1:

                self.log('BUY CREATE @ {}'.format(self.dataclose1[0])) # BUY AT ASK
                order_price = self.dataclose1[0]
                stop = order_price * (1 - self.EMERGENCY_STOP)

                self.order = self.buy(price=order_price, exectype=bt.Order.Market, transmit=True, size=100)

                # self.order_refs.append(self.order)

                # Keep track of the 30m bar id to prevent 2 executions on the same bar (aka waits until another setuo
                # comes rather than executing on the setup again in case it hits the stop/target early
                self.bar_id = len(self.data0)

            elif pred == -1:

                self.log('SELL CREATE @ {}'.format(self.dataclose0[0]))  # SELL AT BID
                order_price = self.dataclose0[0]
                stop = order_price * (1 + self.EMERGENCY_STOP)
                self.order = self.sell(price=order_price, exectype=bt.Order.Market, transmit=True, size=100)

                # self.order_refs.append(self.order)

                # Keep track of the 30m bar id to prevent 2 executions on the same bar (aka waits until another setuo
                # comes rather than executing on the setup again in case it hits the stop/target early
                self.bar_id = len(self.data0)

            else:
                return




def parse_args():

    parser = argparse.ArgumentParser(
        description='Pandas test script')

    parser.add_argument('--noheaders', action='store_true', default=False,
                        required=False,
                        help='Do not use header rows')

    parser.add_argument('--noprint', action='store_true', default=False,
                        help='Print the dataframe')

    return parser.parse_args()


def run(args=None):

    cerebro = bt.Cerebro(stdstats=False)
    ibstore = bt.stores.IBStore(host='127.0.0.1', port=7497)

    data_bid = ibstore.getdata(dataname='AAPL-STK-SMART-USD', timeframe=bt.TimeFrame.Minutes, compression=5, what= 'BID', historical=False, tz=pytz.timezone('US/Eastern'))
    data_ask = ibstore.getdata(dataname='AAPL-STK-SMART-USD', timeframe=bt.TimeFrame.Minutes, compression=5, what= 'ASK', historical=False, tz=pytz.timezone('US/Eastern'))
    data_trades = ibstore.getdata(dataname='AAPL-STK-SMART-USD', timeframe=bt.TimeFrame.Minutes, compression=5, what= 'TRADES', historical=False, tz=pytz.timezone('US/Eastern'))

    cerebro.resampledata(data_bid, timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.resampledata(data_ask, timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.resampledata(data_trades, timeframe=bt.TimeFrame.Minutes, compression=5)

    cerebro.broker = ibstore.getbroker()
    cerebro.addstrategy(ML)
    cerebro.run()
    #cerebro.plot(style='candlestick')

if __name__ == '__main__':
    run()


