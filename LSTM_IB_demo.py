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

        # LOADS LSTM MODEL WITH WEIGHTS

        try:

            json_file = open('LSTM_model.json', 'r')
            LSTM_model_json = json_file.read()
            json_file.close()
            self.LSTM_model = model_from_json(LSTM_model_json)
            # load weights into new model
            self.LSTM_model.load_weights("weights.best.hdf5")
            print("LSTM MODEL SUCCESSFULLY LOADED")
            self.LSTM_model.compile(optimizer = 'adam', loss = 'mean_squared_error')
            print("LSTM MODEL SUCCESSFULLY COMPILED")

        except OSError as e:
            print('OS Error: {}'.format(e))
            print('LSTM MODEL FAILED TO LOAD.')
            exit()

        # LOADS SCALERS FOR LSTM MODEL

        try:

            self.total_vol_scaler = pickle.load(open('total_vol_scaler_file.p', 'rb'))
            self.price_scaler = pickle.load(open('price_scaler_file.p', 'rb'))

            print("SCALERS SUCCESSFULLY LOADED")

        except OSError as e:
            print('OS Error: {}'.format(e))
            print('SCALERS FAILED TO LOAD.')
            exit()


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

        self.bar_lookback = 24

        # Total vol used instead = 9 , if both bid and ask vol then  = 10
        # only 5 feautres wn using only either bid or ask info
        self.features = 5

        # keep track of orders
        self.order_refs = []



        # ID to identify 30min bar it executed on
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



        # CHECKS TO EE IF DATA IS LIVE BEFORE CONTINUING
        # IMPLEMENT safe fault for disconnects later

        if not self.data_live:
            return

        # If not enough data points for LSTM model lookback period, then doesn't execute anything
        if len(self.datas[0]) < self.bar_lookback or len(self.datas[1]) < self.bar_lookback:
            return

        # IF ENOUGH DATAPOINTS AND DATA IS LIVE, SCALES DATA USING LOADED SCALERS AND ADDS TO A DATA ARRAY THAT IS RESHAPED TO PROPER
        # SIZE FOR LSTM MODEL

        open_b = np.array(self.dataopen0.get(size=self.bar_lookback+1))
        high_b = np.array(self.datahigh0.get(size=self.bar_lookback+1))
        low_b = np.array(self.datalow0.get(size=self.bar_lookback+1))
        close_b = np.array(self.dataclose0.get(size=self.bar_lookback+1))
        total_vol = np.array(self.total_volume.get(size=self.bar_lookback+1))


        df = pd.DataFrame()
        df['open_b'] = pd.Series(open_b)
        df['high_b'] = pd.Series(high_b)
        df['low_b'] = pd.Series(low_b)
        df['close_b'] = pd.Series(close_b)
        df['total_vol'] = pd.Series(total_vol).pct_change().round(5)

        # Drops row that has the Nan (first row) because of the total_vol pct_chnage (no preious value to compare
        # it to, keep +1 on all bar lookbacks even if not doing pct_change of prices because you are still doing
        # percentage change of total_vol, dropna would remove the first row of the df so need to have te plus 1 to
        # the dummy variables

        df = df.dropna()

        open_b = self.price_scaler.transform(df['open_b'].values.reshape(-1, 1))
        high_b = self.price_scaler.transform(df['high_b'].values.reshape(-1, 1))
        low_b = self.price_scaler.transform(df['low_b'].values.reshape(-1, 1))
        close_b = self.price_scaler.transform(df['close_b'].values.reshape(-1, 1))
        total_vol = self.total_vol_scaler.transform(df['total_vol'].values.reshape(-1, 1))

        data = np.concatenate((open_b,high_b,low_b,close_b,total_vol), axis = 1)

        # RESHAPES DATA LIST TO BE FED TO LSTM MODEL. ARRAY ALREADY SHAPED FROM OLDEST AT START (TOP) AND NEWEST AT
        # BOTTOM
        # MAKE 3D BY PUTTING 1 AT THE START, MUST BE AT THE SART OR ARRAY SIZE CHANGES, 1 means 1 batch, self.bar lookback
        # is number of row values (bars) in batch used for prediction, self.features is number of coloumn values used
        # to make prediction, all values in this case for both rows and coloumns

        data = data.reshape(1, self.bar_lookback, self.features)


        #Cancels any orders that didn't get executed before placing another one

        for order in self.order_refs:
            if order.status in [order.Submitted, order.Accepted]:
                self.cancel(order)


        # FEEDS MODELS

        LSTM_pred = self.LSTM_model.predict(data)
        print(LSTM_pred)

        LSTM_price_inv = self.price_scaler.inverse_transform(LSTM_pred[:,0:4].reshape(-1,1))
        LSTM_vol_inv = self.total_vol_scaler.inverse_transform(LSTM_pred[:, 4].reshape(-1, 1))
        print(LSTM_price_inv[:,:],LSTM_vol_inv[:,:])

        # ALIASES FOR LSTM_PRED VALUES FOR EASY ACCESS

        open_bid = 0
        high_bid = 1
        low_bid = 2
        close_bid = 3
        volume_bid = 4

        # CLOSES POSITION AS SOON AS THE NEXT BAR IS RECEIVED
        # UPDATE TO ONLY CLOSE TRADE IF MODEL PREDICTS REVERSAL, SO WHEN MODEL PRICE PREDICIN CURVE CHANGES DIRECTION,
        # NOT AT TEH OPEN OF THE NEXT BAR, THIS IS BECAUSE MODEL PREICS THE TREND VERY WELL BUT NOT TEH ExACT PRICES

        if self.position:
            if self.position.size > 0: # and close_b[-1,0] < LSTM_pred[:,close_bid]

                self.log('SELL CREATE (EXIT) @ {}'.format(self.dataclose0[0])) # SELL AT BID
                self.order = self.sell(size=self.position.size, transmit=True, exectype=bt.Order.Market)
                self.order_refs.append(self.order)

            elif self.position.size < 0: # and close_b[-1,0] > LSTM_pred[:,close_bid]

                self.log('BUY CREATE (EXIT) @ {}'.format(self.dataclose1[0])) # BUY AT ASK
                self.order = self.buy(size=self.position.size, transmit=True, exectype=bt.Order.Market)
                self.order_refs.append(self.order)

        # BEFORE PLACING TRADES< IT CHECKS TO SEE IF WE ARE IN THE PROPER TRADING HOURS
        # BEFORE 9:30AM, IT WAITS FOR MARKET TO OPEN
        # AFTER 3:45PM, STOPS SCRIPT AND TRADING STOPS AND END OF TRADING DAY APPROACHES

        if self.datas[0].datetime.time() < datetime.time(9, 35):
            # don't operate until the market opens
            return

        # 15:45 = 3:45pm
        elif self.datas[0].datetime.time() >= datetime.time(15, 45):
            print('END OF TRADING DAY APPROACHING. TRADING STOPPED AT ', self.data.datetime.time())
            exit()


        # Check if we are in the market, time lag allows ensures previous open position has sufficient time to close\
        # before entering into another position

        time.sleep(1.5)

        if not self.position: # and self.bar_id != len(self.datas[0])

            if LSTM_pred[:,close_bid] > LSTM_pred[:,open_bid]:

                self.log('BUY CREATE @ {}'.format(self.dataclose1[0])) # BUY AT ASK
                order_price = self.dataclose1[0] + 0.02
                stop = order_price * (1 - self.EMERGENCY_STOP)
                self.order = self.buy(price=order_price, exectype=bt.Order.Limit, transmit=True, size=100)

                self.order_refs.append(self.order)


            elif LSTM_pred[:,close_bid] < LSTM_pred[:,open_bid]:

                self.log('SELL CREATE @ {}'.format(self.dataclose0[0]))  # SELL AT BID
                order_price = self.dataclose0[0] - 0.02
                stop = order_price * (1 + self.EMERGENCY_STOP)
                self.order = self.sell(price=order_price, exectype=bt.Order.Limit, transmit=True, size=100)

                self.order_refs.append(self.order)


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


