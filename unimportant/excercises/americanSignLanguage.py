import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from keras import utils, models, layers, optimizers, losses, Sequential, callbacks

BATCH_SIZE = 100
IMAGE_SHAPE = 64
CLASS_NAMES = ['C', 'D', 'E', 'F', 'H', 'K', 'L', 'O', 'R', ' ', 'U', 'W']

class Network():
    def __init__(self, name, learning_rate=0.001):
        self.model = Sequential([
            layers.Input((IMAGE_SHAPE, IMAGE_SHAPE, 3)),
            layers.Rescaling(scale=1./255),

            layers.Conv2D(64, (6, 6), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((3, 3), strides=2, padding='same'),

            layers.Conv2D(64, (6, 6), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((3, 3), strides=2, padding='same'),
            
            layers.Conv2D(64, (6, 6), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((3, 3), strides=2, padding='same'),
            layers.Dropout(0.5),

            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(len(CLASS_NAMES), activation=tf.nn.softmax)
        ], name)
        self.compile(learning_rate)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    def train(self, dataset, epochs):
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
        return self.model.fit(dataset, epochs=epochs, steps_per_epoch=dataset.cardinality().numpy(), batch_size=BATCH_SIZE, callbacks=[early_stop])

    def trainWithValidation(self, dataset, validation, epochs):
        early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=2)
        return self.model.fit(dataset, epochs=epochs, steps_per_epoch=dataset.cardinality().numpy(),
        batch_size=BATCH_SIZE, validation_batch_size=BATCH_SIZE, validation_data=validation, validation_steps=validation.cardinality().numpy(), callbacks=[early_stop])

    def evaluate(self, dataset):
        return self.model.evaluate(dataset)
    
    def predict(self, features):
        return self.model.predict(features)

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

    def summary(self):
        self.model.summary()

    def save(self, path):
        self.model.save(path)
        print('Model Saved!')

train_dataset = utils.image_dataset_from_directory(
    './personalDatasets/asl_personal/train', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE), validation_split=0.2, seed=220, subset='training'
)
validation_dataset = utils.image_dataset_from_directory(
    './personalDatasets/asl_personal/train', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE), validation_split=0.2, seed=220, subset='validation'
)
# train_dataset = utils.image_dataset_from_directory(
#     './personalDatasets/asl_personal/train', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
# )
# test_dataset = utils.image_dataset_from_directory(
#     './kaggleDatasets/ASL_Dataset/Test', labels='inferred', batch_size=BATCH_SIZE, image_size=(IMAGE_SHAPE, IMAGE_SHAPE),
# )

# def normalize(images, labels):
#     images /= 255
#     return (images, labels)


#==============================================================
sign_model = Network('sign_model')

history = sign_model.trainWithValidation(train_dataset, validation_dataset,5)
# history = sign_model.train(train_dataset, 10)
sign_model.plot_result(history)
# sign_model.evaluate(test_dataset)
sign_model.save('./savedModels/sign_asl_personal02')
#==============================================================

for images, labels in train_dataset.take(1):
    break

# def predictTenData(features):
#     for i, feature in enumerate(features):
#         plt.subplot(2, 5, i + 1)
#         plt.xlabel(labels.numpy()[i])
#         plt.xticks([]), plt.yticks([])
#         plt.imshow(feature)
#         if i + 1 == 9: break
#     plt.show()

# def showTenData(features):
#     for i, feature in enumerate(features):
#         # Casually normalize the data
#         feature /= 255
#         plt.subplot(2, 5, i + 1)
#         plt.xlabel(labels.numpy()[i])
#         plt.xticks([]), plt.yticks([])
#         plt.imshow(feature)
#         if i + 1 == 9: break
#     plt.show()

# showTenData(images)