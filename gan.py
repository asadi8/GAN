import numpy as np
import keras.backend as K
import os,sys
import generator,discriminator,g_d
import utils
from PIL import Image
K.set_image_dim_ordering('th')
STATE_SHAPE=(1,28,28)
def train(STEP_SIZE_DISCRIMINATOR,STEP_SIZE_G_D,BATCH_SIZE,TOTAL_ITERATIONS,NOISE_SIZE,D_UPDATES_PER_G_UPDATE,run):

    X_Y_train,X_train=utils.build_data()
    discriminator_network=discriminator.discriminator_class(STATE_SHAPE,STEP_SIZE_DISCRIMINATOR)
    generator_network = generator.generator_class(STATE_SHAPE,NOISE_SIZE)
    g_d_network = g_d.g_d_network(generator_network , discriminator_network,STEP_SIZE_G_D)

    for iteration_number in range(TOTAL_ITERATIONS):
        print("iteration #:",iteration_number)   
        
        #update discriminator "D_UPDATES_PER_G_UPDATE" times
        for count in range(D_UPDATES_PER_G_UPDATE):
            discriminator_loss,fake_X_Y,real_X_Y=discriminator_network.update(
                NOISE_SIZE,BATCH_SIZE,X_train,X_Y_train,generator_network)

        #update generator once
        g_d_loss=g_d_network.update(NOISE_SIZE,BATCH_SIZE,X_train,discriminator_network)

        #save sampled fake and real images
        utils.save_image(iteration_number,fake_X_Y,"fake","gan-output-"+str(run))
        utils.save_image(iteration_number,real_X_Y,"real","gan-output-"+str(run))

