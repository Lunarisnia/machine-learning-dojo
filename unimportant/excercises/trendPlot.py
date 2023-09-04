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


# where_prac = np.arange(5)
# print(f"Where Var: {(((where_prac + 1) % 2) / 2) < 2}")
# print(f"Where Var: {where_prac}")
# print(f"Where Var: {where_prac < 2}")
# # plt.plot([0, 1, 2, 3, 4], np.where(where_prac < 0.2, np.cos((np.arange(5) / 2) * np.pi), np.exp([1, 5, 12, 3, 10])))
# # plt.show()
# print(f"Modulo Where Prac: ", where_prac % 2)

def seasonal_pattern(season_time):
    return np.where(season_time < .3, 
        np.cos(season_time * 2 * np.pi),
        1 / np.exp(3 * season_time))

def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)

def white_noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level

amplitude = 40
time = np.arange(4 * 365 + 1)
slope = 0.05
baseline = 10
noise = white_noise(time, noise_level=5, seed=30)
series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
series += noise
plot_trend(time, series)