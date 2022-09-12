import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from matplotlib import pyplot as plt
from keras import layers, utils, optimizers, losses, Sequential
IMAGE_RES = 224
BATCH_SIZE = 100

train_dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats-filtered/train', labels='inferred', image_size=(IMAGE_RES, IMAGE_RES), batch_size=BATCH_SIZE)
validation_dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats-filtered/validation', labels='inferred', image_size=(IMAGE_RES, IMAGE_RES), batch_size=BATCH_SIZE)

URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"
feature_extractor = hub.KerasLayer(URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))

for images, labels in train_dataset.take(1):
    break

feature_batch = feature_extractor(images)
print(feature_batch.shape)

# Freeze the model to be untrainable
feature_extractor.trainable = False

model = Sequential([
    feature_extractor,
    layers.Dense(2, activation=tf.nn.softmax)
], 'tflcd')

model.compile(loss=losses.SparseCategoricalCrossentropy(), optimizer=optimizers.Adam(0.001), metrics=['Accuracy'])

history = model.fit(train_dataset, batch_size=BATCH_SIZE, steps_per_epoch=train_dataset.cardinality().numpy(), 
        epochs=30, validation_data=validation_dataset, validation_batch_size=BATCH_SIZE, validation_steps=validation_dataset.cardinality().numpy())

def plot_loss(history):
    # plt.subplot(1,1,1)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(), plt.xlabel('Epoch'), plt.ylabel('Loss'), plt.show()

plot_loss(history)