import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.layers import LSTM, Dense, Dropout


class Discriminator():
    # steps = 20
    def __init__(self, sess, steps, embedding_dim):
        self.sess = sess
        self.input_x = tf.placeholder(tf.float32, [None, steps, embedding_dim], name="input_x")
        self.y_orig = tf.placeholder(tf.float32, [None, 1], name="y_orig")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.predict = self.model()

    def model(self):
        lstm = LSTM(10, name="main_lstm")(self.input_x)
        drop = Dropout(self.dropout_keep_prob)(lstm)
        out = Dense(1)(drop)
        return out

    def loss(self):
        return tf.reduce_sum(tf.log(self.y_orig + 10e-10) + tf.log(1 - self.predict + 10e-10))
