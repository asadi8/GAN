from keras.optimizers import RMSprop
from keras.models import Sequential
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
