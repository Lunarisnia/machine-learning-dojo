import tensorflow as tf
import keras
from keras import layers
from keras import optimizers
from keras import losses
import tensorflow_datasets as tfds

from matplotlib import pyplot as plt
import numpy as np

# loads the datasets and splits it
datasets, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = datasets['train'], datasets['test']

# fetch all the class names
class_names = metadata.features['label'].names

# List the number of examples on each group
print(f'Number of training Examples: {metadata.splits["train"].num_examples}')
print(f'Number of test Examples: {metadata.splits["test"].num_examples}')


# Normalizer functions to bring it down between 0, 1
def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255 # images = images / 255
    return images, labels


# Since data is not normalized (0 - 255) we need to normalized it to
# between (0 - 1) so we will handle smaller number and that's easier.
train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)

# the first time you use the dataset it'll be loaded from the disk
# caching them helps makes it faster
# train_dataset = train_dataset.cache()
# test_dataset = test_dataset.cache()

model = keras.Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

# SparseCategoricalCrossentropy is also always used when classifying something
model.compile(optimizer=optimizers.Adam(0.001), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# We need to set the dataset to shuffle every epochs so that the ai doesnt memorize
# Todo: Test without shuffling
numTrainExamples = metadata.splits['train'].num_examples
numTestExamples = metadata.splits['test'].num_examples
BATCH_SIZE = 32
train_dataset = train_dataset.cache().repeat().shuffle(numTrainExamples).batch(BATCH_SIZE)
test_dataset = test_dataset.cache().batch(BATCH_SIZE)

history = model.fit(train_dataset, epochs=5, steps_per_epoch=np.ceil(numTrainExamples / BATCH_SIZE))
# history = model.fit(train_dataset, epochs=1, steps_per_epoch=10)
print('Training Finished!')

# Evaluate the model accuracy on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=np.ceil(numTestExamples / BATCH_SIZE))
print(f'Accuracy on test dataset: {test_accuracy}')

# Show 10 images and 10 predictions on the x label and the actual name on y label
for test_images, test_labels in test_dataset.take(1):
    # predicted = model.predict(test_images)
    for i, test_image in enumerate(test_images):
        plt.subplot(2, 5, i + 1)
        plt.imshow(test_image, cmap=plt.cm.binary), plt.xticks([]), plt.yticks([])
        prediction = model.predict(np.array([test_image]))
        plt.xlabel(class_names[np.argmax(prediction[0])])
        if i >= 9: break
    for i, test_label in enumerate(test_labels):
        plt.subplot(2, 5, i + 1)
        plt.ylabel(class_names[test_label])
        if i >= 9: break
    break
plt.show()

# =================================================================================================
# Show a single images after reshaping them removing the color dimension
# for image, label in test_dataset.take(1):
#     break
# image = np.reshape(image, (28, 28))

# plot the image
# plt.imshow(image, cmap=plt.cm.binary), plt.colorbar(), plt.grid(False), plt.show()

# Show 25 images from the train dataset
# for i, (image, label) in enumerate(train_dataset.take(25)):
#     image = np.reshape(image, (28, 28))
#     plt.subplot(5, 5, i + 1)
#     plt.imshow(image, cmap=plt.cm.binary)
#     plt.colorbar()
#     plt.grid(False)
#     plt.xlabel(class_names[label])
#     plt.xticks([]), plt.yticks([])
# plt.show()
# =================================================================================================

