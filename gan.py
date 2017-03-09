import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
import os,sys
import scipy
import generator,discriminator,g_d
K.set_image_dim_ordering('th')


def save_image(update,generated_images,fake_or_real):
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(
            str(update)+"-"+fake_or_real+".png")

def combine_images(generated_images):
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[2:]
        image = np.zeros((height*shape[0], width*shape[1]),
                                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
                img[0,-1,:]=-1
                img[0,0,:]=-1
                img[0,:,0]=-1
                img[0,:,-1]=-1
                i = int(index/width)
                j = index % width
                image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                        img[0, :, :]
        return image
def build_data():
        image_names=os.listdir("gray")
        dim1=len(image_names)
        X_Y_train=np.zeros((dim1,1,28,56))
        X_train=np.zeros((dim1,1,28,28))
        add_to=0
        for index,name in enumerate(image_names):
            if index<len(image_names)-1:
                name_x,name_y=image_names[index],image_names[index+1]
                episode_x,episode_y=int(name_x.split("_")[0]),int(name_y.split("_")[0])
                if episode_x==episode_y:
                            x=scipy.misc.imread("gray/"+image_names[index])
                            x=(np.mean(x,axis=2)- 127.5)/127.5
                            X_train[add_to,0,:,:]=x
                            y=scipy.misc.imread("gray/"+image_names[index+1])
                            y=(np.mean(y,axis=2)- 127.5)/127.5
                            x_y=np.concatenate([x,y],axis=1)
                            X_Y_train[add_to,0,:,:]=x_y
                            add_to=add_to+1
        X_Y_train=X_Y_train[0:add_to,0,:,:].reshape(add_to,1,28,56)
        X_train=X_train[0:add_to,0,:,:].reshape(add_to,1,28,28)
        return X_Y_train,X_train

def train(BATCH_SIZE):

    X_Y_train,X_train=build_data()
    discriminator_network=discriminator.discriminator_class(STATE_SHAPE,STEP_SIZE_DISCRIMINATOR)
    generator_network = generator.generator_class(STATE_SHAPE,NOISE_SIZE)
    g_d_network = g_d.g_d_network(generator_network , discriminator_network,STEP_SIZE_G_D)

    for iteration_number in range(TOTAL_ITERATIONS):
        print("iteration #:",iteration_number)   
        #update discriminator
        for count in range(D_UPDATES_PER_G_UPDATE):
            noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
            random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
            random_X=X_train[random_indices_X,:,:,:]
            fake_X_Y = generator_network.model.predict([random_X,noise_matrix], verbose=0)
            random_indices_X_Y=np.random.choice(len(X_Y_train),BATCH_SIZE,replace=False)
            real_X_Y=X_Y_train[random_indices_X_Y,:,:,:]
            input_discriminator=np.concatenate((real_X_Y,fake_X_Y))
            output_discriminator=[1] * BATCH_SIZE + [0] * BATCH_SIZE
            discriminator_loss = discriminator_network.model.train_on_batch(input_discriminator, output_discriminator)
        #update discriminator

        #update generator
        noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
        random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
        random_X=X_train[random_indices_X,:,:,:]
        discriminator_network.model.trainable=False
        input_g_d=[random_X,noise_matrix]
        output_g_d=[1] * BATCH_SIZE
        g_d_loss=g_d_network.model.train_on_batch(input_g_d,output_g_d)
        discriminator_network.model.trainable=True
        #update generator

        save_image(iteration_number,fake_X_Y,"fake")
        save_image(iteration_number,real_X_Y,"real")

STATE_SHAPE=(1,28,28)
STEP_SIZE_DISCRIMINATOR=0.0001
STEP_SIZE_G_D=0.0001
BATCH_SIZE=128
TOTAL_ITERATIONS=10000
NOISE_SIZE=100
D_UPDATES_PER_G_UPDATE=1

train(BATCH_SIZE)
