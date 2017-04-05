import tensorflow as tf
import numpy as np
import network_helpers as nh

def hook_discriminator(inp):
    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(inp, 5, 1, 3*5, 128, conv_stride=2) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])

    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(c2, 500, bias=0)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0.0)
    with tf.variable_scope('fc3'):
        out = nh.fullyConnected(fc2, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return out

def hook_generator(noise):
    def sub_generator(noise):
        with tf.variable_scope('fc1'):
            fc1 = nh.fullyConnected(noise, 128*7*7, bias=0.0)
        fc1 = tf.reshape(fc1, [-1, 7, 7, 128])

        #with tf.variable_scope('fc2'):
        #    fc2 = nh.fullyConnected(fc1, 1000, bias=0.0)
        #    fc2 = tf.reshape(fc2, [-1, 7, 7, 64])
        #with tf.variable_scope('fc3'):
        #    c2 = tf.reshape(nh.fullyConnected(fc2, 28*28*3, rectifier=tf.nn.sigmoid, bias=0.0), [-1, 28, 28, 3])
        with tf.variable_scope('c1'):
            c1 = nh.upConvolution(fc1, 5, 128, 64, bias=0.0)
        with tf.variable_scope('c2'):
            c2 = nh.upConvolution(c1, 5, 64, 3, rectifier=tf.nn.sigmoid, bias=0.0)
        return c2
    res = []
    for i in range(5):
        with tf.variable_scope('sub_generator', reuse=i>0):
            res.append(sub_generator(noise[:, :, i]))
    c2 = tf.concat(3, res)
    return c2

def hook_normal_mapper(X):
    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(X, 5, 1, 3, 128, conv_stride=2) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])

    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(c2, 500, bias=0)
    with tf.variable_scope('fc2'):
        Z = nh.fullyConnected(fc1, 10, rectifier=lambda x: x, bias=0.0)
    return Z

def hook_normal_discriminator(Z):
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(Z, 100, bias=0)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0)
    with tf.variable_scope('fc3'):
        out = nh.fullyConnected(fc2, 1, rectifier=tf.nn.sigmoid, bias=0)
    return out

inp_data = tf.placeholder(tf.float32, [None, 28, 28, 3*5])
inp_noise = tf.placeholder(tf.float32, [None, 10])

with tf.variable_scope('generator'):
    GZ = hook_generator(inp_noise)

with tf.variable_scope('discriminator'):
    DX = hook_discriminator(inp_data)
with tf.variable_scope('discriminator', reuse=True):
    DGZ = hook_discriminator(GZ)

'''with tf.variable_scope('normal_mapper'):
    EX = hook_normal_mapper(inp_data)
with tf.variable_scope('normal_mapper', reuse=True):
    EGZ = hook_normal_mapper(GZ)

with tf.variable_scope('normal_discr'):
    DZ = hook_normal_discriminator(inp_noise)
with tf.variable_scope('normal_discr', reuse=True):
    DEX = hook_normal_discriminator(EX)
with tf.variable_scope('normal_discr', reuse=True):
    DEGZ = hook_normal_discriminator(EGZ)'''





'''normal_discriminator_loss = (-(tf.reduce_mean(tf.log(DZ)) + tf.reduce_mean(tf.log(1 - DEX)))
                             -(tf.reduce_mean(tf.log(DZ)) + tf.reduce_mean(tf.log(1 - DEGZ))))

normal_gen_loss = (-tf.reduce_mean(tf.log(DEX)) - tf.reduce_mean(tf.log(DEGZ)))'''


discriminator_loss = -(tf.reduce_mean(tf.log(DX)) + tf.reduce_mean(tf.log(1 - DGZ)))


generator_loss = -tf.reduce_mean(tf.log(DGZ))


learning_rate = 0.0001
train_gen = tf.train.AdamOptimizer(learning_rate).minimize(generator_loss, var_list=nh.get_vars('generator'))
train_discr = tf.train.AdamOptimizer(learning_rate).minimize(discriminator_loss, var_list=nh.get_vars('discriminator'))
'''train_normal_discr = tf.train.AdamOptimizer(learning_rate).minimize(normal_discriminator_loss, var_list=nh.get_vars('normal_discr'))
train_normal_gen = tf.train.AdamOptimizer(learning_rate).minimize(normal_gen_loss, var_list=nh.get_vars('normal_mapper'))
'''
saver_gen = tf.train.Saver(var_list=nh.get_vars('generator'))
saver_discr = tf.train.Saver(var_list=nh.get_vars('discriminator'))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.initialize_all_variables())

