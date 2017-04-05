import batch as bh
import network
import numpy as np
import cv2

def train_model(num_steps=-1, gen_name='generator', discr_name='discriminator', disp_interval=100, save_interval=10000):
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    #network.saver_gen.restore(network.sess, './'+gen_name+'.ckpt')
    #network.saver_discr.restore(network.sess, './'+discr_name+'.ckpt')
    while i < num_steps:
        gen_loss = '-'
        discr_loss = '-'
        data = bh.get_batch(32)
        noise = bh.get_noise(32, 10)

        [_, normal_discr_loss] = network.sess.run([network.train_normal_discr, network.normal_discriminator_loss], feed_dict={network.inp_data: data,
                                                               network.inp_noise: noise})

        [_, normal_gen_loss] = network.sess.run([network.train_normal_gen, network.normal_gen_loss], feed_dict={
            network.inp_data: data,
            network.inp_noise: noise
        })

        if i % 1 == 0:
            [_, discr_loss] = network.sess.run([network.train_discr, network.discriminator_loss], feed_dict={network.inp_data: data,
                                                               network.inp_noise: noise})

        [_, gen_loss, gen_image] = network.sess.run([network.train_gen, network.generator_loss, network.GZ], feed_dict={
            network.inp_data: data,
            network.inp_noise: noise
        })

        if np.isnan(discr_loss) or np.isnan(gen_loss):
            break

        if i % disp_interval == 0:
            for j in range(5):
                cv2.imwrite('./recent%s.png' % str(j), 255*gen_image[j])
        if i % save_interval == 0:
            network.saver_discr.save(network.sess, './'+discr_name+'.ckpt')
            network.saver_gen.save(network.sess, './'+gen_name+'.ckpt')

        print i, 'norm_discr', normal_discr_loss, 'norm_gen', normal_gen_loss, 'discr', discr_loss, 'gen', gen_loss
        i += 1

train_model()
