import numpy as np
import tensorflow as tf

from matplotlib import pyplot as plt
from keras import layers
from keras import optimizers
from keras import losses
from keras import Sequential

# Maybe divide by 1k to normalize(?) lets try without normalization first
# bedroom = 1000
# kitchen = 2000
# ac = 5000
# tv = 1250


houses = np.array(
    [
        [1, 1, 6],  # 33000, 13000, 61000, 36000, 27000, 24000, 80000, 172000, 320000
        [2, 3, 1],  # 13000
        [1, 5, 10],  # 61000
        [9, 1, 5],  # 36000
        [9, 9, 0],  # 27000
        [2, 6, 2],  # 24000
        [10, 10, 10],  # 80000
        [50, 11, 20],  # 172000
        [30, 20, 50],  # 320000
    ], dtype=float)
house_prices = np.array(
    [
        33000, 13000, 61000, 36000, 27000, 24000, 80000, 172000, 320000
    ], dtype=float)


test_house = np.array([
    [2, 1, 5],  # 29000, 57000, 32000
    [1, 8, 8],  # 57000
    [1, 8, 3],  # 32000
], dtype=float)
test_price = np.array([
    29000, 57000, 32000
], dtype=float)

# Normalizing the numbers
house_prices = house_prices / 1000
test_price = test_price / 1000

model = Sequential([
    layers.Input(shape=(3,)),
    layers.Dense(128),
    layers.Dense(1)
])

model.compile(optimizer=optimizers.Adam(0.1), loss=losses.MeanSquaredError())

history = model.fit(houses, house_prices, epochs=300)

# test_loss = model.evaluate(test_house, test_price)
# print(f'Test loss: {test_loss}')

prediction = model.predict([[1, 1]])  # 9000

# have to be denormalized
print(f"Result: {np.round(prediction[0][0] * 1000) }")
print(f"Raw Result: {prediction[0][0]}")

plt.plot(history.history['loss']), plt.xlabel(
    'Epochs'), plt.ylabel('Loss'), plt.show()
