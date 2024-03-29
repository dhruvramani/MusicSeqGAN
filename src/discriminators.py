import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential

class Discriminator():
    # steps = 20
    def __init__(self, name, batch_size, steps, embedding_dim, dropout):
        self.name = name
        self.batch_size = batch_size
        self.steps = steps
        self.embedding_dim = embedding_dim # One hot vector 
        self.model_run = self.model(dropout)

    def model(self, dropout):
        with tf.name_scope(self.name):
            model = Sequential()
            # TODO : Add embedding layer
            model.add(LSTM(10, input_shape=[self.steps, self.embedding_dim]))
            model.add(Dropout(dropout))
            model.add(Dense(1))
        return model

    def predict(self, input_x):
        return self.model_run(input_x)

    def loss(self, logits):
        labels = tf.concat([tf.ones(shape = [self.batch_size, 1]), tf.zeros(shape=[self.batch_size, 1])], axis=0)
        # ALT :  tf.reduce_sum(tf.log(labels + 10e-10) + tf.log(1 - predictions + 10e-10))
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)), labels
