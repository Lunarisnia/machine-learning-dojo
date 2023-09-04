from keras import utils
from tensorflow import data
import tensorflow as tf

from matplotlib import pyplot as plt

# https://www.tensorflow.org/api_docs/python/tf/keras/utils/image_dataset_from_directory

train_dataset = utils.image_dataset_from_directory(
    './kaggleDatasets/dogs-vs-cats-filtered/train',
    labels='inferred',
    batch_size=6
)

for images, labels in train_dataset.take(1):
    for i, img in enumerate(images):
        img = tf.cast(img, dtype=tf.int32)
        plt.subplot(2, 3, i + 1)
        plt.imshow(img.numpy())
        plt.xlabel(labels.numpy()[i])
        plt.xticks([]), plt.yticks([])

plt.show()

# What i've learned:
# The dataset need to be split to its own folder
# The folder tree should looks like this for (inferred)
# --- main_dir/
#     --- class_a/
#         ...a.image.01.jpg
#     --- class_b/
#         ...b.image.01.jpg

# The order of the class index is determined sorted by name descending.
# ex:
# cats (0) because c < d
# dogs (1)