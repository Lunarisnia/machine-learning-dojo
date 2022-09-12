import numpy as np

arr = np.array([10, 20, 40, 30], np.uint8)
print(f"Max: {arr.max()}")
print(f"Average: {np.average(arr)}")
print(f"Max Index: {arr.argmax()}")

print(f"Find and return the value that is < 30: {arr[arr < 30]}")
# Print the shape of the array (row, col)
# Ex: arr = [[1, 2, 4], [2, 3, 3]], arr.shape would print (2, 3)
print(f"Shape of the array: {arr.shape}")

# Made an array with random number of 3 by 3
randomArr = np.random.rand(3, 3)
print(randomArr)