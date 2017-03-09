from keras.models import Sequential
from keras.layers import Dense, Input, merge, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, RMSprop
import numpy as np

class discriminator_class:
	model=[]
	ACTIVATION="tanh"
	STEP_SIZE_DISCRIMINATOR=0.0001
	def __init__(self,state_shape,STEP_SIZE_DISCRIMINATOR):
		self.STEP_SIZE_DISCRIMINATOR=STEP_SIZE_DISCRIMINATOR
		self.model=self.build(state_shape)

	def build(self,state_shape):
		if len(state_shape)!=3 or state_shape[0]!=1:
			print("error when initializing dicriminator")
			print("I do not understand the type of input")
			sys.exit(1)
		model = Sequential()
		model.add(Convolution2D(
		                                        128, 5, 5,
		                                        border_mode='same',
		                                        input_shape=(state_shape[0],state_shape[1],state_shape[2]*2) ) )
		model.add(Activation(self.ACTIVATION))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Convolution2D(256, 5, 5))
		model.add(Activation(self.ACTIVATION))
		model.add(MaxPooling2D(pool_size=(2, 2)))
		model.add(Flatten())
		model.add(Dense(1024))
		model.add(Activation(self.ACTIVATION))
		model.add(Dense(1))
		model.add(Activation('sigmoid'))
		d_optim = RMSprop(lr=self.STEP_SIZE_DISCRIMINATOR)
		model.compile(loss='binary_crossentropy', optimizer=d_optim)
		return model

	def update(self,NOISE_SIZE,BATCH_SIZE,X_train,X_Y_train,generator_network):
		noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
		random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
		random_X=X_train[random_indices_X,:,:,:]
		fake_X_Y = generator_network.model.predict([random_X,noise_matrix], verbose=0)
		random_indices_X_Y=np.random.choice(len(X_Y_train),BATCH_SIZE,replace=False)
		real_X_Y=X_Y_train[random_indices_X_Y,:,:,:]
		input_discriminator=np.concatenate((real_X_Y,fake_X_Y))
		output_discriminator=[1] * BATCH_SIZE + [0] * BATCH_SIZE
		discriminator_loss = self.model.train_on_batch(input_discriminator, output_discriminator)
		return discriminator_loss,fake_X_Y,real_X_Y