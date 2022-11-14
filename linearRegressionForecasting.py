import numpy as np
import tensorflow as tf
from keras import layers, Sequential, optimizers, losses, callbacks
from matplotlib import pyplot as plt

def plot_series(time, series, format="-", start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], format, label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    if label:
        plt.legend(fontsize=14)
    plt.grid(True)
    
def trend(time, slope=0):
    return slope * time

def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(season_time < 0.4,
                    np.cos(season_time * 2 * np.pi),
                    1 / np.exp(3 * season_time))
  
def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)
    
def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


amplitude = 40
seed = 33
baseline = 10
slope = 0.05
noise_level = 3
time = np.arange(4 * 365 + 1)
noise = white_noise(time, noise_level=noise_level, seed=seed)
series = baseline + trend(time, slope) + seasonality(time, amplitude=amplitude, period=365)

series += noise

split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]

time_val = time[split_time:]

x_val = series[split_time:]

def window_dataset(series, window_size, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.map(lambda w: (w[:-1], w[-1:]))
    ds = ds.shuffle(len(series))
    return ds.batch(batch_size).prefetch(1)


# DOESNT WORK, TRY AND FIND THE SOLUTION BEFORE SUBMITTING TO THE COLAB
def difference_series(series):
    return series[365:] - series[:-365]

# x_train = difference_series(series)[:split_time]
# x_val = difference_series(series)[split_time:]
window_size = 30
train_set = window_dataset(x_train, window_size)
valid_set = window_dataset(x_val, window_size)

model = Sequential([
    layers.Input([window_size]),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
    layers.Dense(1)
])

optimizer = optimizers.SGD(learning_rate=1e-5, momentum=0.9)
model.compile(optimizer, loss=losses.Huber(), metrics=['mae'])
early_stopping = callbacks.EarlyStopping(patience=10)
model.fit(train_set, epochs=500,
            validation_data=valid_set,
            callbacks=[early_stopping])

def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size))
    ds = ds.batch(32).prefetch(1)
    return model.predict(ds)

lin_forecast = model_forecast(
    model,
    series[split_time - window_size:-1],
    window_size)

# print(lin_forecast[:], "ONLY :")
# print(lin_forecast[:, 0], ":, 0")
# print(lin_forecast[0], "0")
lin_forecast = lin_forecast[:, 0]

plot_series(time_val, x_val, label='Series')
plot_series(time_val, lin_forecast, label='Linear Forecasting NN')
plt.show()