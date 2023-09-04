import tensorflow as tf
import pandas as pd
import numpy as np

from keras import utils, layers, losses, optimizers, models, Sequential
from matplotlib import pyplot as plt

# What i've learned:
# The accuracy on test is way higher than the validation test clear sign of overfitting
# 100% Accuracy - 80% on validation accuracy

# Todo: Try out sigmoid and Binary Crossentropy
# Todo: Dont put softmax and use logits on compile like the guide
class Network():
    def __init__(self, name):
        self.model = Sequential([
            layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(150, 150, 3)), # result: 32 convoluted image of size (150, 150)
            layers.MaxPooling2D((2, 2), strides=2), # 32 Image of size (72, 72)

            layers.Conv2D(64, (3, 3), activation=tf.nn.relu), # 64 convoluted image of size (72, 72)
            layers.MaxPooling2D((2, 2), strides=2), # 64 image of size (18, 18)

            layers.Conv2D(128, (3, 3), activation=tf.nn.relu), # 128 convoluted image of size (18, 18)
            layers.MaxPooling2D((2, 2), strides=2), # 128 image of size (9, 9)

            layers.Conv2D(128, (3, 3), activation=tf.nn.relu), # 128 convoluted image of size (9, 9)
            layers.MaxPooling2D((2, 2), strides=2), # 128 image of size (4, 4)

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
    './kaggleDatasets/dogs-vs-cats-filtered/train',
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    color_mode='rgb',
    shuffle=True
)

validation_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/dogs-vs-cats-filtered/validation',
    labels='inferred',
    batch_size=BATCH_SIZE,
    image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
    color_mode='rgb',
    shuffle=True
)

def normalize(images, labels):
    images /= 255
    return (images, labels)

train_dataset = train_dataset.map(normalize)
validation_dataset = validation_dataset.map(normalize)

cats_dogs_cnn = Network('cvd')
cats_dogs_cnn.compile()

history = cats_dogs_cnn.train(train_dataset, validation_dataset, 100)
cats_dogs_cnn.save('./savedModels/catsVsDogsCnn02Normalized')
# cats_dogs_cnn_reloaded = models.load_model('./savedModels/catsVsDogsCnn02Normalized')

# for images, labels in validation_dataset.take(1):
#     images = images[:12]
#     labels = labels[:12]
#     for i, img in enumerate(images):
#         plt.subplot(3, 4, i + 1)
#         plt.imshow(img)
#         plt.xticks([]), plt.yticks([])
#         x = np.array([img])
#         predictions = cats_dogs_cnn_reloaded.predict(x)
#         predictions = np.argmax(predictions.flatten())
#         plt.xlabel(CLASS_NAMES[predictions])
# plt.show()