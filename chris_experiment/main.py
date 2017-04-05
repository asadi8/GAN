import batch as bh
import numpy as np
import cv2

def train_model(num_steps=-1, gen_name='generator', discr_name='discriminator', disp_interval=100, save_interval=10000):
    import network
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    network.saver_gen.restore(network.sess, './'+gen_name+'.ckpt')
    network.saver_discr.restore(network.sess, './'+discr_name+'.ckpt')
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

        if np.isnan(discr_loss) or np.isnan(gen_loss):
            print 'Got NAN... restoring.'
            network.saver_gen.restore(network.sess, './'+gen_name+'.ckpt')
            network.saver_discr.restore(network.sess, './'+discr_name+'.ckpt')
            continue

        if i % disp_interval == 0:
            for j in range(5):
                ii = np.random.randint(0,3)
                cv2.imwrite('./recent%s.png' % str(j), 255*gen_image[j][:, :, ii])
        if i % save_interval == 0:
            network.saver_discr.save(network.sess, './'+discr_name+'.ckpt')
            network.saver_gen.save(network.sess, './'+gen_name+'.ckpt')

        print i, 'discr', discr_loss, 'gen', gen_loss
        i += 1

def train_action_model(num_steps=-1, gen_name='generator_action', discr_name='discriminator_action', disp_interval=100, save_interval=10000):
    import network_action as network
    if num_steps == -1:
        num_steps = np.inf
    i = 1
    network.saver_gen.restore(network.sess, './'+gen_name+'.ckpt')
    network.saver_discr.restore(network.sess, './'+discr_name+'.ckpt')
    while i < num_steps:
        gen_loss = '-'
        discr_loss = '-'
        (old_screens, actions, new_screens) = bh.get_action_batch(32)
        noise = bh.get_noise(32, 10)

        if i % 1 == 0:
            [_, discr_loss] = network.sess.run([network.train_discr, network.discriminator_loss], feed_dict={
                network.inp_data: new_screens,
                network.inp_old_screen: old_screens,
                network.inp_action: actions,
                network.inp_noise: noise})

        [_, gen_loss, gen_image] = network.sess.run([network.train_gen, network.generator_loss, network.GZ], feed_dict={
            network.inp_data: new_screens,
            network.inp_old_screen: old_screens,
            network.inp_action: actions,
            network.inp_noise: noise
        })

        if np.isnan(discr_loss) or np.isnan(gen_loss):
            print 'Got NAN... restoring.'
            if i < save_interval:
                break 
            network.saver_gen.restore(network.sess, './'+gen_name+'.ckpt')
            network.saver_discr.restore(network.sess, './'+discr_name+'.ckpt')
            continue

        if i % disp_interval == 0:
            for j in range(5):
                ii = np.random.randint(0,3)
                cv2.imwrite('./recent%s.png' % str(j), np.concatenate([255*gen_image[j][:, :, 0], 255*old_screens[j][:, :, 0]], axis=1))
        if i % save_interval == 0:
            network.saver_discr.save(network.sess, './'+discr_name+'.ckpt')
            network.saver_gen.save(network.sess, './'+gen_name+'.ckpt')

        print i, 'discr', discr_loss, 'gen', gen_loss
        i += 1


train_action_model()
