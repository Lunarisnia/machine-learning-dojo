import numpy as np
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
plot_series(time, series), plt.show()

# Splitting the dataset into 2 piece, validation and training data
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]

time_val = time[split_time:]
x_val = series[split_time:]