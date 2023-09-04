import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

from matplotlib import pyplot as plt
from keras import utils, layers, optimizers, losses, Sequential


class Network():
    BATCH_SIZE = 100
    IMAGE_SHAPE = 224
    def __init__(self, name, feature_extractor):
        feature_extractor.trainable = False
        self.name = name
        self.feature_extractor = feature_extractor
        self.model = None

    def create(self):
        self.model = Sequential([
            layers.Input((Network.IMAGE_SHAPE, Network.IMAGE_SHAPE, 3)),
            self.feature_extractor,
            layers.Dense(5, activation=tf.nn.softmax)
        ], self.name)
        return self

    def compile(self, learning_rate=0.001):
        self.model.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.SparseCategoricalCrossentropy(), metrics=['Accuracy'])
        return self
        
    def train(self, train_dataset, validation_dataset, epochs):
        return self.model.fit(train_dataset, validation_data=validation_dataset, batch_size=Network.BATCH_SIZE, steps_per_epoch=train_dataset.cardinality().numpy(),
                                validation_steps=validation_dataset.cardinality().numpy(), epochs=epochs, validation_batch_size=Network.BATCH_SIZE)
    
    def summary(self):
        self.model.summary()

    def predict(self, x):
        return self.model.predict(x)


# Todo: Set the network proper, find one that I can use from tfhub and try it out
model = Network('CatsDogsTFL', hub.KerasLayer('https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2', trainable=False))
model = model.create().compile()
model.summary()

train_dataset = utils.image_dataset_from_directory('./kaggleDatasets/flower_photos/train', 
            batch_size=Network.BATCH_SIZE, image_size=(Network.IMAGE_SHAPE, Network.IMAGE_SHAPE))
validation_dataset = utils.image_dataset_from_directory('./kaggleDatasets/flower_photos/validation',
            batch_size=Network.BATCH_SIZE, image_size=(Network.IMAGE_SHAPE, Network.IMAGE_SHAPE))

history = model.train(train_dataset, validation_dataset, 15)

