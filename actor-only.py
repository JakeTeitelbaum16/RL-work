'''
Goal: create a NN model that uses:
        - reinforcement learning
        - recurrent layers
        - online learning
        - differential Sharpe ratio as a means of reward
        (this is just the Sharpe ratio modified so it can be used with online learning)
:/

other notes
- use training/validation data
- want to scale data (between 0-1??)
- add early stopping in validation testing
- online learning?
'''


import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, LSTM, Dropout
from tensorflow.keras.optimisers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import datetime

#environemnt/trader

class env:
    def __init__(self, data):
        #data is input training data?
        self.data = data
        self.reset()

    def reset(self):
        #initializes trader


    def step(self, action, val):
        # returns new position, reward,
        # whether or not state is terminal, and debug info (print statements)
        next_val = val*data[state+1] # the val multiplied by the percent change of the next data point
        reward = 0 # reward is the differential sharpe ratio
        _state = val * next_val

        if action == 1:
            # buy
        elif action == 2:
            # sell
        else:
            # neutral
        return reward, _state



# Model framework; using LSTM layers to implement recurrent learning
    model = Sequential([
        LSTM(128, input_shape=self.input_shape, activation='relu', return_sequences=True),
        Dropout(0.2),
        BatchNormalization(),

        LSTM(128, activation='relu'),
        Dropout(0.2)
        BatchNormalization(),

        Dense(32, activation='relu'),
        Dense(3, activati='softmax'),
    ])
    opt = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(optimizers=opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy']
                  )
    tensorboard = TensorBoard(log_dir='logs/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    filepath =  # filepath of weights... include epoch number and validation accuracy
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc',
                                 verbose=1, save_best_only=True)
    history = model.fit(x_train, y_train, batch_size=1, epochs=100,
                        validation_data=(validation_x, validation_y),
                        callbacks=[tensorboard, checkpoint]))
    # batch size is 1 because online learning is used