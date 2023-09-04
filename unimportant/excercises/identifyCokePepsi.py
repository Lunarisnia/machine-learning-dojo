import pandas as pd
import numpy as np
import tensorflow as tf

from keras import utils, layers, Sequential, optimizers, losses
from matplotlib import pyplot as plt

BATCH_SIZE = 100
IMAGE_SHAPE = 200
CLASS_NAMES = ['Coca-Cola', 'Pepsi']

class Network():
    def __init__(self, name):
        self.model = Sequential([
            layers.Input((200, 200, 3)),
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2,),
            # layers.RandomZoom(0.2, 0.2),
            layers.Rescaling(scale=1./255),

            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), strides=2, padding='same'),
            # layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            # layers.MaxPooling2D((2, 2), strides=2, padding='same'),

            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dropout(0.3),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(2, activation=tf.nn.softmax)
        ], name)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate=learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    
    def summary(self):
        self.model.summary()

    def train(self, train_dataset, validation_dataset, epochs):
        return self.model.fit(train_dataset, epochs=epochs, steps_per_epoch=train_dataset.cardinality().numpy(),
                validation_data=validation_dataset, validation_steps=validation_dataset.cardinality().numpy(), validation_batch_size=BATCH_SIZE)

    def predict(self, features):
        return self.model.predict(features)

    def rescale_pixel_values(self, images):
        return layers.Rescaling(scale=1./255)(images)

    def plot_result(self, history):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.grid(True), plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Losses')
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.grid(True), plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Accuracy')
        plt.show()

    def plot_images(self, images):
        for i, image in enumerate(images):
            plt.subplot(3, 5, i + 1)
            prediction = self.model.predict(np.array([image]))
            image = image / 255
            plt.imshow(image)
            print(prediction)
            plt.ylabel(CLASS_NAMES[np.argmax(prediction.flatten())])
            # plt.xlabel(CLASS_NAMES[labels[i]])
            plt.xticks([]), plt.yticks([])
            if i > 13: break
        plt.show()
    
    def save(self, path):
        self.model.save(path)
        print('Saved')

train_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/pepsi-coke/train', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
)
validation_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/pepsi-coke/validation', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
)
test_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/pepsi-coke/test', labels=None, batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
)

for images in test_dataset.take(1):
    break

model = Network('cokesitify')
model.compile()
history = model.train(train_dataset, validation_dataset, 20)
model.plot_result(history)
# model.save('./savedModels/cokesitify')

# images = model.rescale_pixel_values(images)
model.plot_images(images)
