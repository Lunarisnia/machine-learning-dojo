import enum
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

img = plt.imread('./testingImages/cliveFace.jpg')
grayscaled = img[:, :, :1]

def convolute(grayscaled):
    kernel = [
        [2, 1, 3],
        [1, 2, 1],
        [4, 2, 4]
    ]
    result = 0
    target_col = 3
    target_row = 3
    for i, k_col in enumerate(kernel):
        for j, k_row in enumerate(k_col):
                

convolute(grayscaled)

plt.imshow(grayscaled, cmap='gray'), plt.show()