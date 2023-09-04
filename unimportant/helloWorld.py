import numpy as np

arr = np.array([65, 72, 67, 82, 72, 91, 67, 73, 71, 70, 85, 87, 68, 86, 83, 90, 74, 89, 75, 61, 65, 76, 71, 65, 91, 79, 75, 69, 66, 85, 95, 74, 73, 68, 86, 90, 70, 71, 88, 68])

first = arr[arr<=100]
last = first[first>91]
# print(len(last))
# print(last)
# print(np.unique(arr))

nums = np.array([-10.4, -0.4, 9.6, 19.6])
power = nums * nums * nums * nums
print(power[0] * 12)
print(power[1] * 10)
print(power[2] * 11)
print(power[3] * 1)

num1 = power[0] * 12
num2 = power[1] * 10
num3 = power[2] * 11
num4 = power[3] * 1
print(f"EPSILON: {num1 + num2 + num3 + num4}")
