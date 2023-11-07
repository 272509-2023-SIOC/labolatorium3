# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
from matplotlib import pyplot as plt

def f1(x):
    return np.sin(x)

def f2(x):
    return np.sin(x - 1)

def f3(x):
    return np.sign(np.sin(8 * x))

def function(x, function_id):
    if function_id == 1:
        return f1(x)
    elif function_id == 2:
        return f2(x)
    elif function_id == 3:
        return f3(x)
    else:
        raise ValueError("Niepoprawny identyfikator funkcji")

x = np.linspace(-np.pi, np.pi, 100)

N = len(x)

kernels = [np.ones(3), np.array([1, 2, 1]), np.array([1, 0, -1]), np.array([-1, 0, 1]), np.array([-1, 2, -1])]

point_counts = [N, 2 * N, 4 * N, 10 * N]

for function_id in [1, 2, 3]:
    print(f"Function: {function.__name__}")
    for kernel in kernels:
        print(f"Kernel: {kernel}")
        for point_count in point_counts:
            x_interpolated = np.linspace(-np.pi, np.pi, point_count)
            y_original = function(x_interpolated, function_id)
            y_interpolated = np.convolve(y_original, kernel, mode='same')
            kernel_sum = np.sum(kernel)
            if kernel_sum != 0:
                y_interpolated /= kernel_sum
            mse = np.mean((y_original - y_interpolated) ** 2)
            print(f"Function: {function_id}, Point count: {point_count}, MSE: {mse}")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
