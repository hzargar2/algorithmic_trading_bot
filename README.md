# Alogirthmic Trading Bot using LSTM Neural Network and Random Forest Classifier
 Trading Algorithms for IB using Machine Learning in Python (includes datasets)

This my first personal project I wrote. It involves algorithmic trading bots using machine learning in python. This project is a product of patching code from other projects with my own to create the trading bots. The feature engineering and model creation portions are strictly my own.
 
Dependencies:

numpy
pandas
sklearn
matplotlib
pickle
sys
backtrader
argparse
datetime
keras
time
pytz
ta
seaborn
utils
scipy

Quickstart instructions:
 
Run LSTM_IB_demo.py (LSTM neural network) and RF_IB_demo.py (Random Forest Classifier) with updated host and port configurations for your Interactive Brokers demo account to test out the algorithms. They use the model, weights, and scaler files that are created through the trainers. Run *_TRAINERS.py files to train the models with your own custom configurations and/or see how the models perform on historical data.

