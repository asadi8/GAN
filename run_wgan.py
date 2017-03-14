import wgan
import os


STEP_SIZE_CRITIC=0.00005
STEP_SIZE_G_C=0.00005
BATCH_SIZE=64
TOTAL_ITERATIONS=10000
NOISE_SIZE=500
C_UPDATES_PER_G_UPDATE=10
CLIP_THRESHOLD=0.1


setting="default"
try:
	os.stat("wgan-output-"+str(setting))
except:
	os.mkdir("wgan-output-"+str(setting))
wgan.train(STEP_SIZE_CRITIC,STEP_SIZE_G_C,BATCH_SIZE,TOTAL_ITERATIONS,NOISE_SIZE,C_UPDATES_PER_G_UPDATE,CLIP_THRESHOLD,setting)