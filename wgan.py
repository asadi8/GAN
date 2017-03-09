import numpy as np
import keras.backend as K
import os,sys
import generator,critic,g_c
import utils
K.set_image_dim_ordering('th')
STATE_SHAPE=(1,28,28)
def train(STEP_SIZE_CRITIC,STEP_SIZE_G_C,BATCH_SIZE,TOTAL_ITERATIONS,NOISE_SIZE,C_UPDATES_PER_G_UPDATE,CLIP_THRESHOLD,run):

    X_Y_train,X_train=utils.build_data()
    critic_network=critic.critic_class(STATE_SHAPE,STEP_SIZE_CRITIC,CLIP_THRESHOLD)
    generator_network = generator.generator_class(STATE_SHAPE,NOISE_SIZE)
    g_c_network = g_c.g_c_network(generator_network , critic_network,STEP_SIZE_G_C)

    for iteration_number in range(TOTAL_ITERATIONS):
        print("iteration #:",iteration_number)   
        
        #update critic "C_UPDATES_PER_G_UPDATE" times
        for count in range(C_UPDATES_PER_G_UPDATE):
            critic_loss,fake_X_Y,real_X_Y=critic_network.update(NOISE_SIZE,BATCH_SIZE,X_train,X_Y_train,generator_network)

        #update generator once
        g_c_loss=g_c_network.update(NOISE_SIZE,BATCH_SIZE,X_train,critic_network)

        #save sampled fake and real images
        utils.save_image(iteration_number,fake_X_Y,"fake","wgan-output-"+str(run))
        #utils.save_image(iteration_number,real_X_Y,"real","wgan-output")

STATE_SHAPE=(1,28,28)
STEP_SIZE_CRITIC=0.00005
STEP_SIZE_G_C=0.00005
BATCH_SIZE=64
TOTAL_ITERATIONS=100000
NOISE_SIZE=100
C_UPDATES_PER_G_UPDATE=10
CLIP_THRESHOLD=0.01
