from keras.models import Sequential
from keras.layers import Dense, Input, merge, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, RMSprop
import keras.backend as K
from keras.constraints import *
import numpy as np

def wgan_loss_critic(predictions,labels):
	return - K.mean(predictions*labels)
	#argmax J=argmin - J

class Clip(Constraint):
    """Clips weights to [-c, c].
    # Arguments
        c: Clipping parameter.
    """
    def __init__(self, c=0.01):
        self.c = c

    def __call__(self, p):
        return K.clip(p, -self.c, self.c)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'c': self.c}


class critic_class:
	model=[]
	ACTIVATION="tanh"
	STEP_SIZE_CRITIC=0.00001
	CLIP_THRESHOLD=0.01
	def __init__(self,state_shape,STEP_SIZE_CRITIC,CLIP_THRESHOLD):
		self.STEP_SIZE_CRITIC=STEP_SIZE_CRITIC
		self.CLIP_THRESHOLD=CLIP_THRESHOLD
		self.model=self.build(state_shape,CLIP_THRESHOLD)

	def build(self,state_shape,CLIP_THRESHOLD):
		if len(state_shape)!=3 or state_shape[0]!=1:
			print("error when initializing critic")
			print("I do not understand the type of input")
			sys.exit(1)
		model = Sequential()
		model.add(Convolution2D(
		                                        128, 5, 5,
		                                        border_mode='same',
		                                        input_shape=(state_shape[0],state_shape[1],state_shape[2]*2),W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD) ) )
		model.add(Activation(self.ACTIVATION))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(256, 5, 5,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD)))
		model.add(Activation(self.ACTIVATION))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(1024,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD)))
		model.add(Activation(self.ACTIVATION))
		model.add(Dense(1,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD)))
		model.add(Activation('linear'))
		c_optim = RMSprop(lr=self.STEP_SIZE_CRITIC)
		#model.compile(loss='binary_crossentropy', optimizer=d_optim)
		model.compile(loss=wgan_loss_critic, optimizer=c_optim)
		return model


	def update(self,NOISE_SIZE,BATCH_SIZE,X_train,X_Y_train,generator_network):
		noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
		random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
		random_X=X_train[random_indices_X,:,:,:]
		fake_X_Y = generator_network.model.predict([random_X,noise_matrix], verbose=0)
		random_indices_X_Y=np.random.choice(len(X_Y_train),BATCH_SIZE,replace=False)
		real_X_Y=X_Y_train[random_indices_X_Y,:,:,:]
		input_discriminator=np.concatenate((real_X_Y,fake_X_Y))
		output_discriminator=[1] * BATCH_SIZE + [-1] * BATCH_SIZE
		discriminator_loss = self.model.train_on_batch(input_discriminator, output_discriminator)
		return discriminator_loss,fake_X_Y,real_X_Y