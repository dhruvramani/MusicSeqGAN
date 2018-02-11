import os
import numpy as np
import tensorflow as tf
from dataset import Dataset
from keras import backend as K
from generator import Generator
from discriminators import Discriminator

# TODO
_BATCH_SIZE = 20 
_STEPS = 1000
_ONEHOT_DIM = 256
_DROPOUT = 0.8
_LEARNING_RATE = 0.01
_NO_EPOCH = 100
_NO_BATCH = 40

def train():
    X = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _STEPS, _ONEHOT_DIM])
    Y = tf.placeholder(tf.float32, shape=[_BATCH_SIZE, _STEPS, _ONEHOT_DIM])
    dropout = tf.placeholder(tf.float32)
    
    XYgen = Generator('XYgen', _BATCH_SIZE, _STEPS, _ONEHOT_DIM, _DROPOUT)
    YXgen = Generator('YXgen', _BATCH_SIZE, _STEPS, _ONEHOT_DIM, _DROPOUT)
    YDisc = Discriminator('Ydisc', _BATCH_SIZE, _STEPS, _ONEHOT_DIM, _DROPOUT)
    XDisc = Discriminator('Xdisc', _BATCH_SIZE, _STEPS, _ONEHOT_DIM, _DROPOUT)

    Yfake = XYgen.predict(X)
    Xfake = YXgen.predict(Y)
    Xback = YXgen.predict(Yfake)
    Yback = XYgen.predict(Xfake)

    DiscYfake = YDisc.predict(Yfake)
    DiscXfake = XDisc.predict(Xfake)
    DiscXreal = XDisc.predict(X)
    DiscYreal = YDisc.predict(Y)
    DiscX = tf.concat([DiscXreal, DiscXfake], axis=0)
    DiscY = tf.concat([DiscYreal, DiscYfake], axis=0)
    
    disc1_loss = XDisc.loss(DiscX) 
    disc2_loss = YDisc.loss(DiscY) 

    gen_loss = XYgen.loss(Yfake) + YXgen.loss(Xfake)
    cycle_loss = tf.square(Xback - X) + tf.square(Yback - Y)
    gen_loss += cycle_loss

    tf.summary.scalar("gen_loss", gen_loss)
    tf.summary.scalar("discY_loss", disc2_loss)
    tf.summary.scalar("discX_loss", disc1_loss)

    tvars = tf.trainable_variables()
    dXvar = [var for var in tvars if 'Xdisc' in var.name]
    dYvar = [var for var in tvars if 'Ydisc' in var.name]
    gvar = [var for var in tvars if 'XYgen' in var.name or 'YXgen' in var.name]

    dXtrain = tf.train.AdamOptimizer(_LEARNING_RATE, beta1=0.5).minimize(disc1_loss, var_list=dXvar)
    dYtrain = tf.train.AdamOptimizer(_LEARNING_RATE, beta1=0.5).minimize(disc2_loss, var_list=dYvar)
    gOptim  = tf.train.AdamOptimizer(_LEARNING_RATE, beta1=0.5)

    val_accuracy = 1.0
    gradients = gOptim.compute_gradients(loss=gen_loss, var_list=gvar)
    
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (grad * val_accuracy, var)

    gtrain = gOptim.apply_gradients(gradients)

    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        data = Dataset(_NO_BATCH, _BATCH_SIZE, _STEPS, _ONEHOT_DIM)
        writer = tf.summary.FileWriter("./tensorboard", sess.graph)
        for epoch in range(_NO_EPOCH):
            losses = [0.0, 0.0, 0.0]
            count = 0
            for X_train, Y_train in data.get_batch("train"):
                feed_dict = {X: X_train, Y: Y_train, dropout: _DROPOUT}
                _, g_loss = sess.run([gtrain, gen_loss], feed_dict=feed_dict)
                _, dx_loss = sess.run([dXtrain, disc1_loss], feed_dict=feed_dict)
                summ, _, dy_loss = sess.run([merged, dYtrain, disc2_loss], feed_dict=feed_dict)
                losses[0] += g_loss
                losses[1] += dx_loss
                losses[2] += dy_loss
                writer.add_summary(summ, count)
                print("Gen : {} Dx : {} Dy : {}".format(g_loss, dx_loss, dy_loss))
                print('\r')
                count += 1
            losses = [i/count for i in losses]
            print("Epoch # {} - Gen : {}, Dx : {}, Dy : {}".format(count, losses[0], losses[1], losses[2]))
            saver.save(sess, './checkpoint/model.ckpt', global_step=count)

        sess.close()

train()