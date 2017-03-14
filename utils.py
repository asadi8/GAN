import numpy as np
from PIL import Image
import argparse
import math
import keras.backend as K
import os,sys
import scipy.misc

def save_image(update,generated_images,fake_or_real,folder):
    image = combine_images(generated_images)
    image = image*127.5+127.5
    Image.fromarray(image.astype(np.uint8)).save(folder+"/"+str(update)+"-"+fake_or_real+".png")

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
        for name in image_names:
            (y_epsiode,y_rest)=name.split("_")
            y_timestep=y_rest.split(".")[0]
            try:
                name_y="gray/"+y_epsiode+"_"+str(int(y_timestep)+1)+".png"
                y=scipy.misc.imread(name_y)
                y=(np.mean(y,axis=2)- 127.5)/127.5
            except Exception as e:
                print(name)
                print(str(e))
                #print("terminal state")
                continue
            x=scipy.misc.imread("gray/"+name)
            x=(np.mean(x,axis=2)- 127.5)/127.5
            X_train[add_to,0,:,:]=x
            x_y=np.concatenate([x,y],axis=1)
            X_Y_train[add_to,0,:,:]=x_y
            x_y=x_y*127.5+127.5
            add_to=add_to+1
        X_Y_train=X_Y_train[0:add_to,0,:,:].reshape(add_to,1,28,56)
        X_train=X_train[0:add_to,0,:,:].reshape(add_to,1,28,28)
        return X_Y_train,X_train