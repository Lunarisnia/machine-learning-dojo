import numpy as np
from tensorflow import data

dataset = data.Dataset.range(10)
dataset = dataset.window(5, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(5))
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
# Gradient descent work best when the dataset is IID (Independent Identically Distributed) so we shuffle it so its independent
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.batch(2).prefetch(1)
for x, y in dataset:
    print(x.numpy(), y.numpy())

def window_dataset(series, window_size, batch_size=32):
    ds = data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    ds = ds.shuffle(len(series))
    return ds.batch(batch_size).prefetch(1)