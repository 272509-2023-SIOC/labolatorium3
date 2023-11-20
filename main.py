# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve


def simplesin(x):
    return np.sin(x)

def inverted_sin(x):
    return np.sin(np.power(x, -1))

def sign(x):
    return np.sign(np.sin(8 * x))

def interpolate_function(x, kernel):
    return convolve(x, kernel, mode='same', method='direct') / np.sum(kernel)

kernels = {
    'boxcar': np.ones(3)/3,
    'hamming': np.hamming(3),
    'gaussian': np.array([1, 2, 1])/3,
    'sobel': np.array([-1, 0, 1]),
    'laplacian': np.array([1, -2, 1])
}

N = 100
multipliers = [2, 4, 10]

x_original = np.linspace(-np.pi, np.pi, N)
y_original = simplesin(x_original)

plt.plot(x_original, y_original, label='Original function', color='blue')

inverted_y = inverted_sin(x_original)
sign_y = sign(x_original)


for kernel_name, kernel in kernels.items():
    for multiplier in multipliers:

        x_interpolate = np.linspace(-np.pi, np.pi, N * multiplier)
        y_interpolate = interpolate_function(y_original, kernel)[:len(x_interpolate)]

        mse_original = np.mean((simplesin(x_interpolate) - y_interpolate)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (original function): {mse_original}')

        plt.plot(x_interpolate, y_interpolate, label=f'{kernel_name} kernel, {multiplier}x points (original function)', alpha=0.7)

        y_interpolate_inverted = interpolate_function(inverted_y, kernel)[:len(x_interpolate)]

        mse_inverted = np.mean((inverted_sin(x_interpolate) - y_interpolate_inverted)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (inverted_sin): {mse_inverted}')

        plt.plot(x_interpolate, y_interpolate_inverted, label=f'{kernel_name} kernel, {multiplier}x points (inverted_sin)', alpha=0.7)

        y_interpolate_sign = interpolate_function(sign_y, kernel)[:len(x_interpolate)]

        mse_sign = np.mean((sign(x_interpolate) - y_interpolate_sign)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (sign): {mse_sign}')

        plt.plot(x_interpolate, y_interpolate_sign, label=f'{kernel_name} kernel, {multiplier}x points (sign)', alpha=0.7)

plt.legend()
plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
