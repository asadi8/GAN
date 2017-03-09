import gan
import os


STEP_SIZE_CRITIC=0.00005
STEP_SIZE_G_C=0.00005
BATCH_SIZE=64
TOTAL_ITERATIONS=10000
NOISE_SIZE=100
C_UPDATES_PER_G_UPDATE=10
CLIP_THRESHOLD=0.01

ssc_list=[0.0001,0.00005,0.00001]
ssgc_list=[0.0001,0.00005,0.00001]
bs_list=[16,32,64,128]
ns_list=[10,100,1000]
cupgu_list=[1,5,10,25,50]


setting=0
for ssc in ssc_list:
	for ssgc in ssgc_list:
		for bs in bs_list:
			for ns in ns_list:
				for cupgu in cupgu_list:
						try:
							os.stat("gan-output-"+str(setting))
						except:
							os.mkdir("gan-output-"+str(setting))
						gan.train(ssc,ssgc,bs,TOTAL_ITERATIONS,ns,cupgu,setting)
						setting=setting+1