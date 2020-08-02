# https://github.com/JakeTeitelbaum16/RL-in-financial-market-predictions.git



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

import pandas as pd

spy = pd.read_csv('/Users/jaketeitelbaum/PycharmProjects/DataProcessing/neededSPY.txt',
                 header=None, usecols=[0,4], names=['dateTime', 'price'])

print(spy.head())


'''
Find Q-learning tutorial
- basic... straitforwrd... tutorial...



'''
