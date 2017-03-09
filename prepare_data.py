import numpy
from PIL import Image
import os,sys


for episode in os.listdir('.'):
	try:
		int(episode)
		image_names=os.listdir('.\\'+str(episode))
		print(image_names)
		for img in image_names:
			print(img)
			img_value=Image.open(str(episode)+"\\"+img).convert('LA')
			img_value=img_value.resize((28,28))
			x=numpy.random.randint(4)
			print(x)
			#img_value=img_value.rotate(x*90)
			img_value.save('.\\gray\\'+img)
	except:
		print("is not a folder",episode)
	'''
	img = Image.open('1\\1_0.png').convert('LA')
	img=img.resize((56,56))
	img.save('greyscale.png')
	'''