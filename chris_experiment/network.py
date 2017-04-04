import tensorflow as tf
import numpy as np
import network_helpers as nh

def hook_discriminator(inp):
    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(inp, 5, 1, 1, 32, conv_stride=2) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 32, 64, conv_stride=2) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(c2, 100, bias=0)
    with tf.variable_scope('fc2'):
        out = nh.fullyConnected(fc1, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return out

def hook_generator(noise):
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(noise, 100, bias=0.0)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 7*7*64, bias=0.0)
        fc2 = tf.reshape(fc2, [-1, 7, 7, 64])
    with tf.variable_scope('c1'):
        c1 = nh.upConvolution(fc2, 5, 64, 32, bias=0.0)
    with tf.variable_scope('c2'):
        c2 = nh.upConvolution(c1, 5, 32, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return c2

inp_data = tf.placeholder(tf.float32, [None, 28, 28, 1])
inp_noise = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('generator'):
    GZ = hook_generator(inp_noise)

with tf.variable_scope('discriminator'):
    DX = hook_discriminator(inp_data)
with tf.variable_scope('discriminator', reuse=True):
    DGZ = hook_discriminator(GZ)

generator_loss = -tf.reduce_mean(tf.log(DGZ))
discriminator_loss = -(tf.reduce_mean(tf.log(DX)) + tf.reduce_mean(tf.log(1 - DGZ)))

learning_rate = 0.000001
train_gen = tf.train.AdamOptimizer(10*learning_rate).minimize(generator_loss, var_list=nh.get_vars('generator'))
train_discr = tf.train.AdamOptimizer(learning_rate/50.).minimize(discriminator_loss, var_list=nh.get_vars('discriminator'))

saver_gen = tf.train.Saver(var_list=nh.get_vars('generator'))
saver_discr = tf.train.Saver(var_list=nh.get_vars('discriminator'))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.initialize_all_variables())

