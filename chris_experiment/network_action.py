import tensorflow as tf
import numpy as np
import network_helpers as nh


def process_old_screen_hook(old_screen):
    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(old_screen, 5, 1, 1, 128, conv_stride=2) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])
    with tf.variable_scope('fc1'):
        s_fc1 = nh.fullyConnected(c2, 100, bias=0)
    return s_fc1

def process_action_hook(action):
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(action, 50, bias=0)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0)
    return fc2

def hook_discriminator(inp, processed_old_screen, processed_action):

    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(inp, 5, 1, 1, 128, conv_stride=2) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])

    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(c2, 500, bias=0)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0.0)
    pre_class = tf.concat(1, [fc2, processed_old_screen, processed_action])
    with tf.variable_scope('fc3'):
        fc3 = nh.fullyConnected(pre_class, 100, bias=0)
    with tf.variable_scope('fc4'):
        out = nh.fullyConnected(fc3, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return out

def hook_generator(noise, processed_old_screen, processed_action):
    
    with tf.variable_scope('n_fc1'):
        n_fc1 = nh.fullyConnected(noise, 100, bias=0)

    combined = tf.concat(1, [n_fc1, processed_old_screen, processed_action])

    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(combined, 128*7*7, bias=0.0)
    fc1 = tf.reshape(fc1, [-1, 7, 7, 128])
    with tf.variable_scope('c1'):
        c1 = nh.upConvolution(fc1, 5, 128, 64, bias=0.0)
    with tf.variable_scope('c2'):
        c2 = nh.upConvolution(c1, 5, 64, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return c2

inp_data = tf.placeholder(tf.float32, [None, 28, 28, 1])
inp_noise = tf.placeholder(tf.float32, [None, 10])
inp_old_screen = tf.placeholder(tf.float32, [None, 28, 28, 1])
inp_action = tf.placeholder(tf.float32, [None, 4])

with tf.variable_scope('proc_screen'):
    processed_old_screen = process_old_screen_hook(inp_old_screen)
with tf.variable_scope('proc_action'):
        processed_action = process_action_hook(inp_action)

with tf.variable_scope('generator'):
    GZ = hook_generator(inp_noise, processed_old_screen, processed_action)

with tf.variable_scope('discriminator'):
    DX = hook_discriminator(tf.reshape(inp_data, [-1, 28, 28, 1]), processed_old_screen, processed_action)
with tf.variable_scope('discriminator', reuse=True):
    DGZ = hook_discriminator(GZ, processed_old_screen, processed_action)


discriminator_loss = -(tf.reduce_mean(tf.log(DX)) + tf.reduce_mean(tf.log(1 - DGZ)))


generator_loss = -tf.reduce_mean(tf.log(DGZ))


learning_rate = 0.00005
train_gen = tf.train.AdamOptimizer(learning_rate).minimize(generator_loss, var_list=nh.get_vars('generator'))
train_discr = tf.train.AdamOptimizer(learning_rate).minimize(discriminator_loss, var_list=nh.get_vars('discriminator'))

saver_gen = tf.train.Saver(var_list=nh.get_vars('generator'))
saver_discr = tf.train.Saver(var_list=nh.get_vars('discriminator'))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.initialize_all_variables())

