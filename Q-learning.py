'''
- creating en environment that can be used for reinforcement leanring
- can use other models from
- add early stopping in validation testing


'''

import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimisers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import pandas as pd                 #import from console
import matplotlib.pyplot as plt     #import from console
#import sklearn as skl
import random

hourlySPY = open('/Users/jaketeitelbaum/PycharmProjects/DataProcessing/neededSPY.txt')
hourlyQQQ = open('/Users/jaketeitelbaum/PycharmProjects/DataProcessing/neededQQQ.txt')
hourlyGBP_USD = open('/Users/jaketeitelbaum/PycharmProjects/DataProcessing/hourlyGBP-USD.txt')
# close statements at end of file

'''
Data Preprocessing
--------------------
NN learn better w compressed data
- can change data points to % change compared to previous data pt
then use sklearn train_test_split(hourlySPY, test_size=0.2, shuffle=False)
- dont want to shuffle data bc the sequence of the data is important to leaning TIME SERIES data
- then each state is very similar to last and can lead to repeat actions...
'''




EPSILON = 0.3
#EPSILON_DECAY = 0.995
LEARNING_RATE = 0.001
LR_DECAY =
DISCOUNT = 0.99
TRANSACTION_COST = 0.001 # percent

# do I need action memory here?...
class env:
    def __init__(self, data, num_states=4, num_actions=3):
        #data is input training data?
        self.data = hourlySPY
        self.reset()
        self.num_states = num_states
        self.num_actions = num_actions


        #initializing Q-table
        self.Q = np.zeros((num_states, num_actions))




    def reset(self):
        #initializes trader

    def action(self, s):
        """
        Returns an action using epsilon-greedy
        """
        # Exploration
        if (np.random.uniform(0.0, 1.0) < EPSILON):
            a = np.random.randint(0, self.num_actions)

        # Exploitation
        else:
            a = np.argmax(self.Q[s, :])

        # After each update reduce the chance of exploration

        EPSILON = EPSILON * EPSILON_DECAY

        return a



    def step(self, val):    # returns new position, reward,
                            # whether or not state is terminal, and debug info (print statements)

        # explorations vs exploitations
        rand = np.random.random()
        if rand < EPSILON:
            action = int((np.random.random() * 3) + 1)


        next_val = val*data[state+1] # the val multiplied by the percent change of the next data point
        reward = 0 # reward is the differential sharpe ratio
        # --> cant be calculated on first step so must take rand action at start?

        # Differential Sharpe ratio found in part 4.3 of this paper - http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.87.8437&rep=rep1&type=pdf
        # code is here - https://github.com/AchillesJJ/DSR


        _state = val * next_val

        current_q = # current Q value
        max_q = # action associated w max reward

        # Q-learning ???
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_q)
        # how to calculate new q value...

        if action == 1:
            # buy,
        elif action == 2:
            # sell
        else:
            # neutral
        return reward, _state





class agent:
    def __init__(self, gamma=0.99,
                 input_shape,):
        # self.everything above

        self.gamma = gamma
        self.input_shape = input_shape


    def DQN(self):
        model = Sequential([
            Dense(128, input_shape=self.input_shape, activation='relu'),
            BatchNormalization(),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dense(3, activation='softmax')
        ])
        opt = Adam(learning_rate=0.001, decay=1e-6)
        model.compile(optimizers=opt,
                      loss='mse', # loss can also use (sparse) categorical crossentropy
                      metrics=['accuracy']
        )

        tensorboard = TensorBoard(log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        filepath = #filepath of weights... include epoch number and validation accuracy
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                     verbose=1, save_best_only=True)
        history = model.fit(x_train, y_train, batch_size=64, epochs=100,
                                validation_data=(validation_x, validation_y),
                                callbacks=(tensorboard, checkpoint), verbose=1)



        #use tensorboard and matplotlib to analyze data?
        #want to see validation loss/accuracy and model loss/accuracy



hourlySPY.close()
hourlyQQQ.close()
hourlyGBP_USD.close()
