import tensorflow as tf
import numpy as np
import PIL.Image as Image
import tensorflow_hub as hub

from matplotlib import pyplot as plt
from keras import layers, optimizers, losses, Sequential, utils

CLASSIFIER_URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
IMAGE_RES = 224
BATCH_SIZE = 10

model = Sequential([
    hub.KerasLayer(CLASSIFIER_URL, input_shape=(IMAGE_RES, IMAGE_RES, 3))
], 'tfLearningCatsDogs')


grace_hopper = utils.get_file('image.jpg', 'https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize((IMAGE_RES, IMAGE_RES))

grace_hopper = np.array(grace_hopper)/255

result = model.predict(grace_hopper[np.newaxis, ...])
print(result.shape)

# Todo: Decode the predictions by downloading the labels
labels_path = utils.get_file('ImageNetLabels.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

# Test the prediction with the decoded labels
# plt.imshow(grace_hopper), plt.axis('off'), plt.title(imagenet_labels[np.argmax(result.flatten())])
# plt.show()
# =============================================

train_dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats-filtered/train', labels='inferred', image_size=(IMAGE_RES, IMAGE_RES), batch_size=BATCH_SIZE)
validation_dataset = utils.image_dataset_from_directory('./kaggleDatasets/dogs-vs-cats-filtered/validation', labels='inferred', image_size=(IMAGE_RES, IMAGE_RES), batch_size=BATCH_SIZE)

normalizer = layers.Rescaling(scale=1./255)

for images, labels in train_dataset.take(1):
    images = images.numpy() / 255
    break

raw_predictions = model.predict(images)

for i, image in enumerate(images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(image), plt.title(imagenet_labels[np.argmax(raw_predictions[i])])
plt.show()
    

