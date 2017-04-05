import cv2
import numpy as np
import os


dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../gray')

images = [cv2.imread(os.path.join(dataset_path, x)) for x in os.listdir(dataset_path)
          if x.endswith('.png')]
images = np.array(images)

def get_batch(batch_size):
    batch = []
    for i in range(batch_size):
        indices = np.random.randint(0, len(images), size=5)
        print images.shape
        sub_batch = np.transpose(images[indices, :, :, [0]], [1, 2, 3, 0])
        batch.append(sub_batch)
    return np.array(batch) / 255.

def get_noise(batch_size, noise_size):
    return np.random.normal(size=(batch_size, noise_size, 5))

print get_batch(32).shape
