from keras.models import Model
from keras.layers import Dense, Input, merge, Reshape
from keras.layers.core import Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import SGD, RMSprop
import keras.backend as K
import sys

class generator_class:
	model=[]
	NOISE_SIZE=100
	ACTIVATION_FUNCTION="tanh"
	# add leaky
	def __init__(self,input_shape,NOISE_SIZE):
		self.NOISE_SIZE=NOISE_SIZE
		self.model=self.build_convolutional(input_shape)
		print(self.model.summary())

	def build_convolutional(self,state_shape):
		state=Input(shape=state_shape, name='state')
		z=Input(shape=(self.NOISE_SIZE,),name="noise")

		y=Convolution2D(32, 5, 5,border_mode='same')(state)
		y=Activation(self.ACTIVATION_FUNCTION)(y)
		y=MaxPooling2D(pool_size=(2, 2))(y)
		y=Convolution2D(32, 5, 5,border_mode='same')(y)
		y=Activation(self.ACTIVATION_FUNCTION)(y)
		y=MaxPooling2D(pool_size=(2, 2))(y)
		y=Flatten()(y)
		y_z=merge([y,z],mode="concat")

		x=Dense(512)(y_z)
		x=Activation(self.ACTIVATION_FUNCTION)(x)
		x=Dense(8*7*7)(x)
		#x=BatchNormalization()(x)
		x=Activation(self.ACTIVATION_FUNCTION)(x)
		x=Reshape((8, 7, 7), input_shape=(8*7*7,))(x)
		x=UpSampling2D(size=(2, 2))(x)
		x=Convolution2D(16, 5, 5, border_mode='same')(x)
		x=Activation(self.ACTIVATION_FUNCTION)(x)
		x=UpSampling2D(size=(2, 2))(x)
		x=Convolution2D(1, 5, 5, border_mode='same')(x)
		x=Activation('tanh')(x)
		output=merge([state,x],mode="concat")
		return Model(input=[state, z], output=[output])

	def build_dense(self,state_shape):
		state=Input(shape=state_shape, name='state')
		z=Input(shape=(self.NOISE_SIZE,),name="noise")
		y=Flatten()(state)
		y=Dense(1024)(y)
		y=Activation(self.ACTIVATION_FUNCTION)(y)
		y_z=merge([y,z],mode="concat")

		x=Dense(128*7*7)(y_z)
		x=Activation(self.ACTIVATION_FUNCTION)(x)
		x=Dense(1*28*28)(y_z)
		x=Activation('tanh')(x)
		x=Reshape((1, 28, 28))(x)

		output=merge([state,x],mode="concat")
		return Model(input=[state, z], output=[output])

