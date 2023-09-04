import tensorflow as tf
import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
from keras import utils, models, layers, optimizers, losses, Sequential

CLASS_NAMES = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

class Network():
    def __init__(self, name, learning_rate=0.001):
        self.model = Sequential([
            layers.Input((28, 28, 1)),
            layers.RandomFlip('horizontal'),
            layers.RandomZoom(0.2, 0.2, fill_mode='constant'),
            layers.Rescaling(scale=1./255),

            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), strides=2, padding='same'),
            layers.Dropout(0.3),
            
            layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu),
            layers.MaxPooling2D((2, 2), strides=2, padding='same'),
            layers.Dropout(0.3),

            layers.Flatten(),
            layers.Dense(512, activation=tf.nn.relu),
            layers.Dense(26, activation=tf.nn.softmax)
        ], name)
        self.compile(learning_rate)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    def train(self, features, labels, epochs):
        return self.model.fit(features, labels, epochs=epochs, validation_split=0.2)

    def evaluate(self, features, labels):
        return self.model.evaluate(features, labels)
    
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

raw_test_dataset = pd.read_csv('./kaggleDatasets/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv')
raw_train_dataset = pd.read_csv('./kaggleDatasets/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv')

test_features = raw_test_dataset.iloc[:, 1:]
test_labels = raw_test_dataset.iloc[:, 0]

train_features = raw_train_dataset.iloc[:, 1:]
train_labels = raw_train_dataset.iloc[:, 0]

# Reshape the data because its flatenned
test_features = test_features.values.reshape(-1, 28, 28, 1)
test_labels = test_labels.values.reshape(-1,)

train_features = train_features.values.reshape(-1, 28, 28, 1)
train_labels = train_labels.values.reshape(-1,)

#==============================================================
# sign_model = Network('sign_model')
# history = sign_model.train(train_features, train_labels, 8)
# sign_model.plot_result(history)
# sign_model.evaluate(test_features, test_labels)
# sign_model.save('./savedModels/sign_mnistValidationSplit')
#==============================================================

sign_model_reloaded = models.load_model('./savedModels/sign_mnist01')
ten_predictions = sign_model_reloaded.predict(test_features[:10])

for i, feature in enumerate(test_features[:10]):
    plt.subplot(2, 5, i + 1), plt.imshow(feature, cmap='gray')
    plt.xticks([]), plt.yticks([])
    predict = np.argmax(ten_predictions[i])
    answer = test_labels[i]
    plt.xlabel(CLASS_NAMES[predict], color='blue' if predict == answer else 'red')
plt.show()

# color='blue' if lbl[i].numpy() == np.argmax(predicted[i]) else 'red'

# def showTenData(features):
#     for i, feature in enumerate(features[:10]):
#         # Casually normalize the data
#         feature = np.float64(feature)
#         feature /= 255
#         plt.subplot(2, 5, i + 1)
#         plt.imshow(feature, cmap='gray')
#     plt.show()

# showTenData(train_features)