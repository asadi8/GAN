from keras.optimizers import RMSprop
from keras.models import Sequential
import numpy as np
class g_d_network:
	model=[]
	STEP_SIZE_G_D=0.0001
	def __init__(self,generator,discriminator,STEP_SIZE_G_D):
		self.STEP_SIZE_G_D=STEP_SIZE_G_D
		self.model=self.build(generator,discriminator)

	def build(self,generator,discriminator):
		model = Sequential()
		model.add(generator.model)
		model.add(discriminator.model)
		g_d_optim = RMSprop(lr=self.STEP_SIZE_G_D)
		model.compile(loss='binary_crossentropy', optimizer=g_d_optim)
		return model
	def update(self,NOISE_SIZE,BATCH_SIZE,X_train,discriminator_network):
		noise_matrix=np.random.normal(0,1,NOISE_SIZE*BATCH_SIZE).reshape(BATCH_SIZE,NOISE_SIZE)
		random_indices_X=np.random.choice(len(X_train),BATCH_SIZE,replace=False)
		random_X=X_train[random_indices_X,:,:,:]
		discriminator_network.model.trainable=False
		input_g_d=[random_X,noise_matrix]
		print("value of fake before GAN update:",np.mean(self.model.predict(input_g_d)))
		output_g_d=[1] * BATCH_SIZE
		g_d_loss=self.model.train_on_batch(input_g_d,output_g_d)
		print("value of fake after GAN update:",np.mean(self.model.predict(input_g_d)))
		discriminator_network.model.trainable=True
		return g_d_loss
