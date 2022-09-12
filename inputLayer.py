import tensorflow as tf
from tensorflow import keras
from keras import layers

model = keras.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(2, activation='relu', name='layer1')
    ]
)

model.summary()