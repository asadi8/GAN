import gan
import os


STEP_SIZE_CRITIC=0.00005
STEP_SIZE_G_C=0.00005
BATCH_SIZE=64
TOTAL_ITERATIONS=10000
NOISE_SIZE=100
C_UPDATES_PER_G_UPDATE=1


setting="default"
try:
	os.stat("gan-output-"+str(setting))
except:
	os.mkdir("gan-output-"+str(setting))
gan.train(STEP_SIZE_CRITIC,STEP_SIZE_G_C,BATCH_SIZE,TOTAL_ITERATIONS,NOISE_SIZE,C_UPDATES_PER_G_UPDATE,setting)
setting=setting+1