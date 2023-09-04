import os
import tensorflow as tf
import pandas as pd
import numpy as np

from keras import utils, layers, losses, optimizers, models, Sequential
from matplotlib import pyplot as plt

# Goal: Learn how ot use dropout layer and image augmentation to prevent overfitting

class Network():
    def __init__(self, name):
        self.model = Sequential([
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2),
            layers.RandomZoom(0.2, 0.2),
            layers.Rescaling(scale=1./255),

            layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)), # result: 32 convoluted image of size (150, 150)
            layers.MaxPooling2D((2, 2), strides=2), # 32 Image of size (72, 72)

            layers.Conv2D(64, (3, 3), activation=tf.nn.relu), # 64 convoluted image of size (72, 72)
            layers.MaxPooling2D((2, 2), strides=2), # 64 image of size (18, 18)

            layers.Conv2D(128, (3, 3), activation=tf.nn.relu), # 128 convoluted image of size (18, 18)
            layers.MaxPooling2D((2, 2), strides=2), # 128 image of size (9, 9)

            layers.Conv2D(128, (3, 3), activation=tf.nn.relu), # 128 convoluted image of size (9, 9)
            layers.MaxPooling2D((2, 2), strides=2), # 128 image of size (4, 4)

            layers.Dropout(0.5), # 50% Chance of the values coming into the dropout layer will be dropped
            layers.Flatten(), # Flatten all 128 image
            layers.Dense(512, activation=tf.nn.relu), # usual hidden dense layer of 512 neuron
            layers.Dense(2, activation=tf.nn.softmax) # Output layer using softmax of either dog or cat
        ], name)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    def summary(self):
        self.model.summary()

    def train(self, train_dataset, validation_dataset, epochs):
        return self.model.fit(train_dataset, epochs=epochs, steps_per_epoch=train_dataset.cardinality().numpy(),
                 validation_data=validation_dataset, validation_steps=validation_dataset.cardinality().numpy())

    def save(self, path):
        self.model.save(path)
        print(f'Saved to: {path}')

    def evaluate(self, x, y):
        self.model.evaluate(x, y, batch_size=BATCH_SIZE)
    
    def predict(self, x):
        self.model.predict(x, batch_size=BATCH_SIZE)

BATCH_SIZE = 100 # Number of training examples to process before updating our models variables
IMAGE_SHAPE = 150 # Set the training data to consist of 150 x 150 image
CLASS_NAMES = ['Cat', 'Dog']

train_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/dogs-vs-cats/train',
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    color_mode='rgb',
    shuffle=True
)

validation_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/dogs-vs-cats/validation',
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    color_mode='rgb',
    shuffle=True
)

cats_dogs_cnn = Network('cvd')
cats_dogs_cnn.compile()

history = cats_dogs_cnn.train(train_dataset, validation_dataset, 100)
# cats_dogs_cnn.save('./savedModels/catsVsDogsCnnFull')

# ======================================================================================================
# cats_dogs_cnn_reloaded = models.load_model('./savedModels/catsVsDogsCnnFull')
# labels = []
# test_dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats/test', labels=None, batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE), shuffle=False)

# file_paths = test_dataset.file_paths
# for file_path in file_paths:
#     labels.append(int(file_path.split('\\')[1].split('.')[0]))

# labels = np.array(labels)
# predictions = cats_dogs_cnn_reloaded.predict(test_dataset)

# datas = []
# for i, pred in enumerate(predictions):
#     datas.append((labels[i], np.argmax(pred)))
# dataframe = pd.DataFrame(datas, columns=['id', 'label'])
# dataframe = dataframe.sort_values(by=['id'])
# dataframe.to_csv('./cat-vs-dogs-submission.csv', index=False)

# for images in test_dataset.take(1):
#     images = images[:12]
#     predictions = cats_dogs_cnn_reloaded.predict(images)
#     for i, img in enumerate(images):
#         plt.subplot(3, 4, i + 1)
#         plt.imshow(img.numpy() / 255)
#         plt.xticks([]), plt.yticks([])
#         x = np.array([img])
#         # predictions = cats_dogs_cnn_reloaded.predict(x)
#         # predictions = np.argmax(predictions.flatten())
#         plt.xlabel(CLASS_NAMES[np.argmax(predictions[i])])
# plt.show()