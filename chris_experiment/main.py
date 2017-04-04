import batch as bh
import network
import numpy as np
import cv2

def train_model(num_steps=-1, gen_name='generator', discr_name='discriminator', disp_interval=100, save_interval=10000):
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    while i < num_steps:
        gen_loss = '-'
        discr_loss = '-'
        data = bh.get_batch(32)
        noise = bh.get_noise(32, 10)
        if i % 1 == 0:
            [_, discr_loss] = network.sess.run([network.train_discr, network.discriminator_loss], feed_dict={network.inp_data: data,
                                                               network.inp_noise: noise})

        [_, gen_loss, gen_image] = network.sess.run([network.train_gen, network.generator_loss, network.GZ], feed_dict={
            network.inp_data: data,
            network.inp_noise: noise
        })

        if i % disp_interval == 0:
            cv2.imwrite('./recent.png', 255*gen_image[0])
        if i % save_interval == 0:
            network.saver_discr.save(network.sess, './'+discr_name+'.ckpt')
            network.saver_gen.save(network.sess, './'+gen_name+'.ckpt')

        print i, 'discr', discr_loss, 'gen', gen_loss
        i += 1

train_model()
