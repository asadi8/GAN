from keras.models import Sequential, Model
from keras.layers import Dense, Input, merge, Reshape, Lambda
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
		state_state_prime=Input(shape=(state_shape[0],state_shape[1],state_shape[2]*2))
		state=Lambda(lambda x: x[:,:,:,0:state_shape[2]],output_shape=state_shape)(state_state_prime)
		state_prime=Lambda(lambda x: x[:,:,:,state_shape[2]:state_shape[2]*2],output_shape=state_shape)(state_state_prime)
		difference=merge([state, state_prime], mode=lambda x: x[0] - x[1], output_shape=lambda x: x[0])
		
		image = Input(shape=state_shape)
		x = Convolution2D(32, 5, 5,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD))(image)
		x=Activation(self.ACTIVATION)(x)
		x = Convolution2D(32, 5, 5,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD))(x)
		x=Activation(self.ACTIVATION)(x)
		out = Flatten()(x)
		vision_model = Model(image, out)



		state=vision_model(state)
		state_prime=vision_model(state_prime)
		difference=vision_model(difference)

		merged=merge([state,state_prime,difference],mode="concat")
		merged=Dense(1024,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD))(merged)
		merged=Activation(self.ACTIVATION)(merged)
		merged=Dense(128,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD))(merged)
		merged=Activation(self.ACTIVATION)(merged)
		merged=Dense(1,W_constraint=Clip(CLIP_THRESHOLD),b_constraint=Clip(CLIP_THRESHOLD))(merged)
		merged=Activation('linear')(merged)

		model=Model(input=[state_state_prime],output=[merged])
		c_optim = RMSprop(lr=self.STEP_SIZE_CRITIC)
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
		real_score_before=np.mean(self.model.predict(real_X_Y))
		fake_score_before=np.mean(self.model.predict(fake_X_Y))
		discriminator_loss = self.model.train_on_batch(input_discriminator, output_discriminator)
		real_score_after=np.mean(self.model.predict(real_X_Y))
		fake_score_after=np.mean(self.model.predict(fake_X_Y))
		print(real_score_after)
		print(fake_score_after)
		print("ws distance after critic update:",real_score_after-fake_score_after)
		return discriminator_loss,fake_X_Y,real_X_Y