import tensorflow as tf
from keras import layers
from keras import Sequential
from keras import optimizers
import numpy as np
from matplotlib import pyplot as plt

celcius_q = np.array([0, 3, 9, 1, 80, 19, 100], dtype=float)
fahrenheit_a = np.array([32, 37, 48, 34, 176, 66, 212], dtype=float)

# for i, c in enumerate(celcius_q):
#     print(f"{c} degree celcius is equal {fahrenheit_a[i]} degree fahrenheit.")

model = Sequential(
    [
        layers.Input(1),
        # layers.Dense(4, name='layer0'),
        # layers.Dense(2, name='layer1'),
        layers.Dense(1, name='layer2')
    ]
)

# Model compile is the act of compiling the model before training them with .fit()
model.compile(loss='mean_squared_error', optimizer=optimizers.Adam(0.1))

# Model fit is the act of actually training the model
# model.fit(feature, label, epochs, verbose)
history = model.fit(celcius_q, fahrenheit_a, epochs=1000)
print('Training finished!')

# showing the loss magnitude over time in matplotlib
plt.xlabel('Epoch number')
plt.ylabel('Loss magnitude')
plt.plot(history.history['loss']), plt.show()

# use the trained model to predict
print(model.predict([150.0]))

# look at the layer weights
print(f"These are the layer variables: {model.layers[0].get_weights()}")
#array([[1.8042316]], dtype=float32)
# array([31.634102], dtype=float32)