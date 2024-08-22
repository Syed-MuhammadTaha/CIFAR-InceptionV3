import pickle
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def unpickle(file):
    with open(file, 'rb') as f:
        return pickle.load(f, encoding='bytes')
    
b1 = unpickle("cifar-10-batches-py/data_batch_1")
meta = unpickle("cifar-10-batches-py/batches.meta")

data = b1[b'data']
labels = b1[b'labels']
meta = meta[b'label_names']
print(meta)


# save the 10000 image in folders named according to their labels and the name should increment in their folders
for i in range(10):
    os.mkdir(meta[i].decode("utf-8"))


class_count = np.zeros(10, dtype=int)
for i in range(10000):
    
    img = data[i].reshape(3, 32, 32).transpose([1, 2, 0])
    label = labels[i]
    folder = meta[label].decode("utf-8")
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(f"{folder}/{class_count[label]}.png", img)
    class_count[label] += 1








