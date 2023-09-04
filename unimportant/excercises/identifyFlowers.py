import numpy as np
import pandas as pd
import tensorflow as tf

from matplotlib import pyplot as plt
from keras import utils, models, layers, Sequential, optimizers, losses

BATCH_SIZE = 100
IMAGE_SHAPE = 150

class Network():
    def __init__(self, name):
        self.model = Sequential([
            layers.Input((150, 150, 3)),
            layers.RandomFlip('horizontal'),
            layers.RandomRotation(0.2, fill_mode='constant'),
            layers.RandomZoom(0.15, 0.15, fill_mode='constant'),
            layers.Rescaling(scale=1./255),

            # layers.Conv2D(16, (3, 3), activation=tf.nn.relu),
            # layers.MaxPooling2D((2, 2), strides=2),
            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), padding='same', strides=2),
            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), padding='same', strides=2),
            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), padding='same', strides=2),

            layers.Dropout(0.5),
            layers.Flatten(),
            # layers.Dense(256, activation=tf.nn.relu),
            layers.Dense(256, activation=tf.nn.relu),
            layers.Dropout(0.2),
            layers.Dense(5, activation=tf.nn.softmax)
        ], name)
    
    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    def train(self, train_dataset, validation_dataset, epochs=50):
        return self.model.fit(train_dataset, batch_size=BATCH_SIZE, steps_per_epoch=train_dataset.cardinality().numpy(),
                epochs=epochs, validation_data=validation_dataset, validation_batch_size=BATCH_SIZE, validation_steps=validation_dataset.cardinality().numpy())
    
    def plot_result(self, history):
        plt.subplot(1, 2, 1)
        plt.plot(history.history['loss'], label='losses')
        plt.plot(history.history['val_loss'], label='val_losses')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save(self, path):
        self.model.save(path)
        print(f"Saved to: {path}")

train_dataset = utils.image_dataset_from_directory('./kaggleDatasets/flower_photos/train',
     batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE), 
     labels='inferred')
validation_dataset = utils.image_dataset_from_directory('./kaggleDatasets/flower_photos/validation',
     batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE), 
     labels='inferred')

rescale = layers.Rescaling(scale=1./255)

def plot_images(images, labels):
    for i, img in enumerate(images):
        plt.subplot(5, 10, i + 1)
        img = rescale(img)
        plt.imshow(img)
        plt.xlabel(labels[i].numpy())
        plt.xticks([]), plt.yticks([])
    plt.show()

model = Network('IdentifyFlowers')
model.compile()

history = model.train(train_dataset, validation_dataset, 80)
model.save('./savedModels/identifyFlowersOwn')
model.plot_result(history)
