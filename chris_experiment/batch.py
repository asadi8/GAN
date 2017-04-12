import cv2
import numpy as np
import os
import re


dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../gray')
action_dataset_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../raw_dataset')
images = [cv2.imread(os.path.join(dataset_path, x))[:, :, [0]] for x in os.listdir(dataset_path)
          if x.endswith('.png')]
images = np.array(images)

def open_action_dataset():
    print action_dataset_path
    episodes = os.listdir(action_dataset_path)
    episodes_data = []
    for ep in episodes:
        ep_path = os.path.join(action_dataset_path, ep)
        dirs = [x for x in os.listdir(ep_path) if x.endswith('.png')]
        ordered_dirs = sorted(dirs, key=lambda x: int(re.match(r'^\d+\_(\d+)\.png$', x).groups()[0]))
        ordered_screens = np.array([np.reshape(np.sum(cv2.resize(cv2.imread(os.path.join(ep_path, d)), (28, 28)), axis=2)/3., [28, 28, 1])
                                    for d in ordered_dirs])

        with open(os.path.join(ep_path, 'actions.txt'), 'r') as f:
            actions = np.array([eval(x) for x in f.readlines()])
        old_screens = ordered_screens[:-1]
        new_screens = ordered_screens[1:]
        episodes_data.append((old_screens, new_screens, actions))
    return episodes_data

action_dataset = open_action_dataset()

def get_batch(batch_size):
    batch = []
    indices = np.random.randint(0, len(images), batch_size)
    batch = np.reshape(images[indices], [batch_size, 28, 28, 1])
    return batch 
    #for i in range(batch_size):
        #indices = np.random.randint(0, len(images), size=3)
        #sub_batch = np.transpose(images[indices], [1, 2, 3, 0])
        #batch.append(sub_batch)
    #return np.reshape(np.array(batch) / 255., [batch_size, 28, 28, 3])

def get_action_batch(batch_size):
    old_screens = []
    actions = []
    new_screens = []
    for i in range(batch_size):
        ep = np.random.randint(0, len(action_dataset))
        i = np.random.randint(0, len(action_dataset[ep][0]))
        old_screens.append(action_dataset[ep][0][i])
        new_screens.append(action_dataset[ep][1][i])
        actions.append(action_dataset[ep][2][i])
    return np.array(old_screens)/255., np.array(actions), np.array(new_screens)/255.


def get_noise(batch_size, noise_size):
    return np.random.normal(size=(batch_size, noise_size))

#print get_batch(32).shape
