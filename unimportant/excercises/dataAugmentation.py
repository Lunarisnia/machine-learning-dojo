from matplotlib import pyplot as plt
import numpy as np
from keras import layers, Sequential, utils

# Goal: Learn how to Data Augmentation
# What i've learned augmenting image is convenient

BATCH_SIZE = 5
IMG_SHAPE = 250

dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats-filtered/train', image_size=(IMG_SHAPE, IMG_SHAPE), batch_size=BATCH_SIZE)

resize_and_rescale = Sequential([
    layers.Resizing(150, 150),
    layers.Rescaling(scale=1./255),
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(1, 1),
    layers.RandomCrop(100, 100)
])

for images, labels in dataset.take(1):
    break

result = resize_and_rescale(images[0])
result2 = resize_and_rescale(images[0])
result3 = resize_and_rescale(images[0])

plt.subplot(2, 3, 1)
plt.imshow(result)
plt.subplot(2, 3, 2)
plt.imshow(result2)
plt.subplot(2, 3, 3)
plt.imshow(result3)
plt.xticks([]), plt.yticks([])
plt.show()