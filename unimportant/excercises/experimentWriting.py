import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
from keras import layers
from keras import optimizers
from keras import losses
from keras import Sequential
from keras import utils


datasets, metadata = tfds.load('mnist', as_supervised=True, with_info=True)
train, test = datasets['train'], datasets['test']
train_set_num, test_set_num = metadata.splits['train'].num_examples, metadata.splits['test'].num_examples

class_label = metadata.features['label'].names

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train = train.map(normalize)
test = test.map(normalize)

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=optimizers.Adam(0.001), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

BATCH_SIZE = 32
train = train.cache().repeat().shuffle(train_set_num).batch(BATCH_SIZE)
test = test.cache().batch(BATCH_SIZE)

history = model.fit(train, epochs=5, steps_per_epoch=np.ceil(train_set_num / BATCH_SIZE))
test_loss, test_acc = model.evaluate(test, steps=np.ceil(test_set_num / BATCH_SIZE))

myNine = utils.load_img('./testingImages/four01.png', keep_aspect_ratio=True)
myNineArr = np.array(myNine, dtype=float)
myNineNormed = np.array([myNineArr[:, :, :1] / 255])
prediction = model.predict(myNineNormed)
plt.imshow(myNineNormed[0], cmap=plt.cm.binary), plt.xlabel(np.argmax(prediction[0])), plt.show()

x = test.take(32)

for i, (img, lbl) in enumerate(x):
    predicted = model.predict(img)
    plt.subplot(8, 4, i + 1)
    plt.imshow(img[i], cmap=plt.cm.binary)
    plt.ylabel(np.argmax(predicted[i]), color='blue' if lbl[i].numpy() == np.argmax(predicted[i]) else 'red')
    plt.xticks([]), plt.yticks([])
plt.show()