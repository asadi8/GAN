from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np
import keras.backend as K

def wgan_loss_g_c(predictions,labels):
	return - K.mean(predictions*labels)
	#argmax J=argmin - J

class g_c_network:
	model=[]
	STEP_SIZE_G_C=0.00001
	def __init__(self,generator,critic,STEP_SIZE_G_C):
		self.STEP_SIZE_G_C=STEP_SIZE_G_C
		self.model=self.build(generator,critic)

	def build(self,generator,critic):
		model = Sequential()
		model.add(generator.model)
		model.add(critic.model)
		g_c_optim = RMSprop(lr=self.STEP_SIZE_G_C)
		model.compile(loss=wgan_loss_g_c, optimizer=g_c_optim)
		return model

	def update(self,NOISE_SIZE,BATCH_SIZE,X_train,critic_network):
		noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
		random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
		random_X=X_train[random_indices_X,:,:,:]
		critic_network.model.trainable=False
		input_g_c=[random_X,noise_matrix]
		output_g_c=[1] * BATCH_SIZE
		print("value before wGAN update:",np.mean(self.model.predict(input_g_c)))
		g_c_loss=self.model.train_on_batch(input_g_c,output_g_c)
		print("value after wGAN update:",np.mean(self.model.predict(input_g_c)))
		critic_network.model.trainable=True
		return g_c_loss
