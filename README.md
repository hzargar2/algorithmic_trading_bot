# trade_algos
 Trading Algorithms for IB using Machine Learning in Python (includes datasets)
 
These are some of my personal projects involving algorithmic trading bots using machine learning in python. It involves patching together some code from other public projets with my own code. In addition, the feature engineering portions are strickly my own code.
 
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
 
Run LSTM_IB_demo.py (LSTM neural network) and RF_IB_demo.py (Random Forest Classifier) with updated host and port configurations for your Interactive Brokers demo account to test out the algorithms. They use the model, weights, and scaler files that are created through the trainers. Run *TRAINERS.py files to train the models with yur own custom configurations and/or see how the models perform on historical data.

