import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from matplotlib import pyplot as plt
from keras import utils, layers, optimizers, losses, Sequential


class Network():
    BATCH_SIZE = 100
    def __init__(self, name, feature_extractor):
        self.model = Sequential([
            feature_extractor,
            layers.Dense(5, activation=tf.nn.softmax)
        ], name)

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['Accuracy'])
        
    def train(self, train_dataset, validation_dataset, epochs):
        return self.model.fit(train_dataset, validation_data=validation_dataset, batch_size=Network.BATCH_SIZE, steps_per_epoch=train_dataset.cardinality().numpy(),
                                validation_steps=validation_dataset.cardinality().numpy(), epochs=epochs, validation_batch_size=Network.BATCH_SIZE)
    
    def predict(self, x):
        return self.model.predict(x)


model = Network('Aa', layers.Dense(1))

# Todo: Set the network proper, find one that I can use from tfhub and try it out