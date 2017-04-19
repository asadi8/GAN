import tensorflow as tf
import numpy as np
import network_helpers as nh

def hook_discriminator(inp):
    with tf.variable_scope('c1'):
        c1 = nh.downConvolution(inp, 5, 1, 1, 128, conv_stride=2, rectifier=tf.nn.elu) # 14 x 14 x 32
    with tf.variable_scope('c2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2, rectifier=tf.nn.elu) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])

    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(c2, 500, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0.0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc3'):
        out = nh.fullyConnected(fc2, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    return out

def discriminator_autoencoder(inp, specified_encoding=None):
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(tf.reshape(inp, [-1, 28*28]), 500, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 100, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc3'):
        fc3 = nh.fullyConnected(fc2, 500, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc4'):
        fc4 = tf.reshape(nh.fullyConnected(fc3, 28*28, bias=0, rectifier=tf.nn.sigmoid), [-1, 28, 28, 1])
    return fc4
    '''print inp
    if specified_encoding is None:
        with tf.variable_scope('c1'):
            c1 = nh.downConvolution(inp, 5, 1, 1, 10, conv_stride=2, rectifier=tf.nn.elu) # 14 x 14 x 32
        with tf.variable_scope('c2'):
            c2 = nh.downConvolution(c1, 5, 1, 10, 20, conv_stride=2, rectifier=tf.nn.elu) # 7 x 7 x 64
            c2 = tf.reshape(c2, [-1, 7*7*20])
        with tf.variable_scope('fc1'):
            fc1 = nh.fullyConnected(c2, 100, bias=0, rectifier=tf.nn.elu)
    else:
        fc1 = specified_encoding
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 20*7*7, bias=0.0, rectifier=tf.nn.elu)
    fc2 = tf.reshape(fc2, [-1, 7, 7, 20])
    with tf.variable_scope('dc1'):
        c1 = nh.upConvolution(fc2, 5, 20, 10, bias=0.0, rectifier=tf.nn.elu)
    with tf.variable_scope('dc2'):
        c2 = nh.upConvolution(c1, 5, 10, 1, rectifier=tf.nn.sigmoid, bias=0.0)'''

    return c2

def hook_generator(noise):
    with tf.variable_scope('fc1'):
        fc1 = nh.fullyConnected(noise, 100, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc2'):
        fc2 = nh.fullyConnected(fc1, 500, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('fc3'):
        fc3 = nh.fullyConnected(fc2, 28*28, bias=0, rectifier=tf.nn.sigmoid)
        image = tf.reshape(fc3, [-1, 28, 28, 1])
    #with tf.variable_scope('fc4'):
    #    fc4 = nh.fullyConnected(fc3, 500, bias=0, rectifier=tf.nn.elu)
    #with tf.variable_scope('fc5'):
    #    fc5 = nh.fullyConnected(fc4, 100, bias=0, rectifier=tf.nn.elu)
    #with tf.variable_scope('fc6'):
    #    recon_noise = nh.fullyConnected(fc5, 10, bias=0, rectifier=lambda x: x)
    return image#, recon_noise
    '''with tf.variable_scope('enc'):
        fc1 = nh.fullyConnected(noise, 64*7*7, bias=0.0, rectifier=tf.nn.elu)
    fc1 = tf.reshape(fc1, [-1, 7, 7, 64])
    with tf.variable_scope('c1'):
        c1 = nh.upConvolution(fc1, 5, 64, 32, bias=0.0, rectifier=tf.nn.elu)
    with tf.variable_scope('c2'):
        image = nh.upConvolution(c1, 5, 32, 1, rectifier=tf.nn.sigmoid, bias=0.0)
    with tf.variable_scope('dc1'):
        c1 = nh.downConvolution(image, 5, 1, 1, 128, conv_stride=2, rectifier=tf.nn.elu) # 14 x 14 x 32
    with tf.variable_scope('dc2'):
        c2 = nh.downConvolution(c1, 5, 1, 128, 64, conv_stride=2, rectifier=tf.nn.elu) # 7 x 7 x 64
        c2 = tf.reshape(c2, [-1, 7*7*64])
    with tf.variable_scope('dfc1'):
        fc1 = nh.fullyConnected(c2, 500, bias=0, rectifier=tf.nn.elu)
    with tf.variable_scope('dfc2'):
        recon_noise = fc2 = nh.fullyConnected(fc1, 10, bias=0.0, rectifier=lambda x: x)
    return image, recon_noise'''



inp_data = tf.placeholder(tf.float32, [None, 28, 28, 1])
inp_noise = tf.placeholder(tf.float32, [None, 10])
inp_k = tf.placeholder(tf.float32)
inp_lambda = 0.001
gamma = 0.5
with tf.variable_scope('generator'):
    GZ = hook_generator(inp_noise)

with tf.variable_scope('discriminator'):
    DX = discriminator_autoencoder(inp_data)

with tf.variable_scope('discriminator', reuse=True):
    print GZ
    DGZ = discriminator_autoencoder(GZ)


#with tf.variable_scope('discriminator'):
#    DX = hook_discriminator(tf.reshape(inp_data, [-1, 28, 28, 3]))
#with tf.variable_scope('discriminator', reuse=True):
#    DGZ = hook_discriminator(GZ)

def L(x, xhat):
    return tf.reduce_mean(tf.square(x - xhat))

LX = L(inp_data, DX)
LGZ = L(DGZ, GZ)


discriminator_loss =  LX - inp_k * LGZ
generator_loss = LGZ
loss = discriminator_loss + generator_loss
new_k = tf.clip_by_value(inp_k + inp_lambda*(gamma*LX - LGZ), 0, 1)
#discriminator_loss = -(tf.reduce_mean(tf.log(DX)) + tf.reduce_mean(tf.log(1 - DGZ)))


#generator_loss = -tf.reduce_mean(tf.log(DGZ))


learning_rate = 0.00001
train_gen = tf.train.AdamOptimizer(learning_rate).minimize(generator_loss, var_list=nh.get_vars('generator'))
train_discr = tf.train.AdamOptimizer(learning_rate).minimize(discriminator_loss, var_list=nh.get_vars('discriminator'))

saver_gen = tf.train.Saver(var_list=nh.get_vars('generator'))
saver_discr = tf.train.Saver(var_list=nh.get_vars('discriminator'))

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

sess.run(tf.initialize_all_variables())

