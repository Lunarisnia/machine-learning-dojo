import tensorflow as tf
import pandas as pd

d = { 'col1': [1, 2, 5, 6, 8], 'col2': [3, 4, 8, 1, 2], 'col3': [5, 6, 9, 1, 4], 'total': [9, 12, 22, 8, 14] }
df = pd.DataFrame(d)
print(df)

dataset = tf.data.Dataset.from_tensor_slices(df)
dataset = dataset.window(3, shift=1)
dataset = dataset.flat_map(lambda w: w.batch(3))
dataset = dataset.map(lambda w: (w[1:], w[:-1]))
# dataset = dataset.shuffle(2)
print(list(dataset.as_numpy_iterator()))