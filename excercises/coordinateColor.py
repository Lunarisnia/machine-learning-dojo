import tensorflow as tf
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from keras import layers, optimizers, losses, Sequential, models

# [4, 0, 'Blue'],
# [1, 1, 'Red'],
# [2, 3, 'Blue'],
# [-4, 4, 'Red'],
# [0, -4, 'White'],
# [-4, 3, 'Red'],
# [-3, 3, 'Red'],
# [-2, -3, 'Red'],
# [4, 4, 'Blue'],
# [0, 1, 'White'],
# [0, 0, 'White'],
# [3, 2, 'Blue'],
# [-1, -4, 'Red'] 

class_names = ['Right', 'Left', 'Neutral']
# Todo: Maybe Hotmap this
dataset = pd.DataFrame(np.array([
    [4, 0, 0],
    [1, 1, 0],
    [2, 3, 0],
    [0, 4, 2],
    [-4, 4, 1],
    [0, -4, 2],
    [0, 0, 2],
    [-4, 3, 1],
    [-3, 3, 1],
    [-2, -3, 1],
    [0, 7, 2],
    [4, 4, 0],
    [0, 6, 2],
    [0, 8, 2],
    [0, 1, 2],
    [0, 0, 2],
    [3, 2, 0],
    [0, 3, 2],
    [-1, -4, 1],
    [-5, 6, 1],
    [1, 8, 0],
    [1, 5, 0],
    [7, -6, 0],
    [2, 8, 0],
    [-5, -7, 1],
    [-8, -4, 1],
    [-5, -1, 1],
    [6, -6, 0],
    [-4, 7, 1],
    [3, -2, 0],
    [-6, -6, 1],
    [0, -6, 2],
    [0, -5, 2]
]), columns=['X', 'Y', 'Color'])

# Maybe we dont need to hotmap this??
# dataset['Color'] = dataset['Color'].map({0: 'Blue', 1: 'Red', 2: 'White'})
# dataset = pd.get_dummies(dataset, columns=['Color'], prefix='', prefix_sep='')

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# print(test_dataset.count())
# exit()
train_features = train_dataset.copy()
train_labels = train_features.pop('Color')

test_features = test_dataset.copy()
test_labels = test_features.pop('Color')

class Model:
    def __init__(self, name):
        self.model = Sequential([
            layers.Input((2,)),
            layers.Dense(16, activation=tf.nn.relu),
            layers.Dense(32, activation=tf.nn.relu),
            layers.Dense(3, activation=tf.nn.softmax)
        ], name)
    
    def compile(self, learning_rate):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

    def train(self, features, labels, epochs):
        return self.model.fit(features, labels, epochs=epochs)
    
    def evaluate(self, test_features, test_labels):
        return self.model.evaluate(test_features, test_labels)

    def predict(self, x):
        return self.model.predict(x)

    def plot_accuracy(self, history):
        plt.plot(history.history['accuracy'])
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.show()
    
    def plot_prediction(self, predictions, labels):
        predictions = np.array(list(map(lambda x: np.argmax(x), predictions)))
        labels = labels.to_numpy()
        plt.scatter(predictions, labels)
        lims = [-8, 8]
        plt.xlim(lims)
        plt.ylim(lims)
        plt.plot(lims, lims)
        plt.show()

    def save(self, path):
        self.model.save(path)
        print(f'Model Saved to: {path}')

model = Model('Predictor')
model.compile(0.09)
history = model.train(train_features, train_labels, 10)
# model.plot_accuracy(history)
model.evaluate(test_features, test_labels)

predictions = model.predict(test_features)

# Todo: Need to find a way to better plot this
# Scatter doesnt work because the point is either 0, 1, 2
# model.plot_prediction(predictions, test_labels)

# Testing Predictions
while 1:
    raw_input = input('Enter a matrix (x,y): ')
    matrix = np.array([[int(raw_input.split(',')[0]), int(raw_input.split(',')[1])]])
    pred = class_names[np.argmax(model.predict(matrix).flatten())]
    print(f'{matrix.flatten()} is on the {pred} side')


