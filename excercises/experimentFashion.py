import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from matplotlib import pyplot as plt
from keras import layers
from keras import optimizers
from keras import losses
from keras import Sequential

datasets, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = datasets['train'], datasets['test']

class_names = metadata.features['label'].names

model = Sequential([
    layers.Flatten(input_shape=(28, 28, 1)),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=optimizers.Adam(0.001), loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels

train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


BATCH_SIZE = 32
num_of_train_dataset = metadata.splits['train'].num_examples
num_of_test_dataset = metadata.splits['test'].num_examples
train_dataset = train_dataset.cache().repeat().shuffle(num_of_train_dataset).batch(BATCH_SIZE)


history = model.fit(train_dataset, epochs=5, steps_per_epoch=np.ceil(num_of_train_dataset / BATCH_SIZE))

test_dataset = test_dataset.cache().batch(BATCH_SIZE)
test_loss, test_acc = model.evaluate(test_dataset, steps=np.ceil(num_of_test_dataset / BATCH_SIZE))
print(f"Test Acc: {test_acc}")



# 1. Don't normalize the pixel values and see the effect that has
# Ans: The accuracy of the test deteriorates quite significantly from 89% to about 77%

# 2. Set training epochs to 1
# Ans: It'll train only for one epoch and it effects the accuracy because the model doens't have enough training

# 3. Change the number of neurons on the hidden dense layer to 512
# Ans: The test took longer to finish, the accuracy starts higher than the one with fewer neurons, and the end training results has higher accuracy as well.
# But weirdly the evaluation result is the same as the one with 128 neuron

# 4. Add an additional hidden layer
# Ans: more layer of neuron equal more processing power needed to train and it afflicts the training result

# Number of neurons and layers definitely determine the end result and more doesnt always equal better.