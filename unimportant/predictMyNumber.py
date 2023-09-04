import keras
import tensorflow as tf
from keras import layers
from keras import losses
from keras import optimizers
from matplotlib import pyplot as plt
import numpy as np

feature = np.array([3, 1, 10, 5, 9, 11, 2, 6], dtype=int) # x * 3 + 7
label = np.array([16, 10, 37, 22, 34, 40, 13, 25], dtype=int)
# feature = np.array([3, 1, 10], dtype=float) # ((x + 10) / 2) * 2
# label = np.array([13, 11, 20], dtype=float)
# feature = np.array([5, 2], dtype=float) # x + 32
# label = np.array([57, 54], dtype=float)


# feature = np.array([90, 1, 5], dtype=float) # x° × π/180
# label = np.array([1.571, 0.01745, 0.08727], dtype=float)

# feature = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], dtype=float) # x + 10, x - 2, x * 2 alternating every 2number
# label = np.array([11, 12, 1, 2, 10, 12, 17, 18, 7, 8, 22, 24, 23, 24, 13, 14, 34, 36], dtype=float) # unsolved

model = keras.Sequential([
    layers.Input(1),
    layers.Dense(8),
    layers.Dense(16),
    layers.Dense(1)
], "MyPredictor")

model.compile(loss=losses.MSE, optimizer=optimizers.Adam(0.1))

trainingHistory = model.fit(feature, label, epochs=500)
print('Training Done!')

plt.xlabel('epochs'), plt.ylabel('Loss'), plt.plot(trainingHistory.history['loss']), plt.show()

print(model.layers[0].get_weights())

while 1:
    num = int(input('Input the number to predict: '))
    print(f"{num} is {np.round(model.predict([num])[0][0])} according to your 'SECRET' formula.")
    # print(f"{num} is {model.predict([num])[0][0]} according to your 'SECRET' formula.")