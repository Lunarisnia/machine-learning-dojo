import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

from matplotlib import pyplot as plt
from keras import layers
from keras import optimizers
from keras import losses
from keras import Sequential

datasets, metadata = tfds.load(
    'fashion_mnist', as_supervised=True, with_info=True)
train_dataset, test_dataset = datasets['train'], datasets['test']

class_names = metadata.features['label'].names
# Todo: Test without 2 layer of convolutions: Require more training since it doesnt overfit as bad as 2conv layer
# Todo: Test without pooling the image: It overfit really bad
# Todo: Test without activation on the conv layer: Bad and overfit worst too
# Todo: Test without padding: Same as the normal but lower accuracy
model = Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), strides=2, padding='same'),
    layers.Conv2D(64, (3, 3), activation=tf.nn.relu),
    layers.MaxPooling2D((2, 2), strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=optimizers.Adam(0.001),
              loss=losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])


def normalize(images, labels):
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


train_dataset = train_dataset.map(normalize)
test_dataset = test_dataset.map(normalize)


BATCH_SIZE = 32
num_of_train_dataset = metadata.splits['train'].num_examples
num_of_test_dataset = metadata.splits['test'].num_examples
train_dataset = train_dataset.cache().repeat().shuffle(
    num_of_train_dataset).batch(BATCH_SIZE)


history = model.fit(train_dataset, epochs=10, steps_per_epoch=np.ceil(
    num_of_train_dataset / BATCH_SIZE))

test_dataset = test_dataset.cache().batch(BATCH_SIZE)
test_loss, test_acc = model.evaluate(
    test_dataset, steps=np.ceil(num_of_test_dataset / BATCH_SIZE))
print(f"Test Acc: {test_acc}")

print(history.history)

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