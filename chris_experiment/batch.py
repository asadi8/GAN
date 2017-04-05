import cv2
import numpy as np
import os


dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../gray')

images = [cv2.imread(os.path.join(dataset_path, x))[:, :, [0]] for x in os.listdir(dataset_path)
          if x.endswith('.png')]
images = np.array(images)

def get_batch(batch_size):
    batch = []
    for i in range(batch_size):
        indices = np.random.randint(0, len(images), size=1)
        sub_batch = np.transpose(images[indices], [1, 2, 3, 0])
        batch.append(sub_batch)
    print np.array(batch).shape 
    return np.reshape(np.array(batch) / 255., [batch_size, 28, 28, 3])

def get_noise(batch_size, noise_size):
    return np.random.normal(size=(batch_size, noise_size))

print get_batch(32).shape
