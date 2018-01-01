import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class Generator(object):
    def __init__(self, batch_size, steps, embedding_dim, dropout):
        self.name = name
        self.batch_size = batch_size
        self.steps = step
        self.embedding_dim = embedding_dim
        self.model_run = self.model(dropout)

    def model(self, dropout):
        with tf.name_scope(self.name):
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=(self.steps, self.embedding_dim)))
            model.add(LSTM(32, return_sequences=True))
            model.add(Dense(self.embedding_dim, activation="softmax"))
        return model

    def predict(self, input):
        return self.model_run(input)

    def loss(self, predictons):
        # Implement like discriminator_on_generator_loss
