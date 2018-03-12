import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

class Generator(object):
    def __init__(self, name, batch_size, steps, embedding_dim, dropout):
        self.name = name
        self.batch_size = batch_size
        self.steps = steps
        self.embedding_dim = embedding_dim
        self.model_run = self.model(dropout)

    def model(self, dropout):
        with tf.name_scope(self.name):
            model = Sequential()
            model.add(LSTM(32, return_sequences=True, input_shape=(self.steps, self.embedding_dim)))
            model.add(Dropout(dropout))
            model.add(LSTM(32, return_sequences=True))
            model.add(Dropout(dropout))
            model.add(Dense(self.embedding_dim, activation="softmax"))
        return model

    def predict(self, input):
        return self.model_run(input)

    def loss(self, predictions):
        return tf.reduce_mean(tf.log(predictions))
        #labels = tf.concat([tf.ones(shape = [self.batch_size, 1]), tf.zeros(shape=[self.batch_size, 1])], axis=0)
        # ALT :  tf.reduce_sum(tf.log(labels + 10e-10) + tf.log(1 - predictions + 10e-10))
        #return tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions)
