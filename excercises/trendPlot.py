from cProfile import label
from matplotlib import pyplot as plt
import numpy as np;


def trend(time, slope=0):
    return time * slope


def plot_trend(time, series, start=0, end=None, label=None):
    plt.plot(time[start:end], series[start:end], label=label)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid()
    plt.show()

time = np.arange(4 * 30)
series = trend(time, 0.2)

plot_trend(time, series)