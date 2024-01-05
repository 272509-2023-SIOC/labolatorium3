# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# Definicje funkcji
def simplesin(x):
    return np.sin(x)

def inverted_sin(x):
    return np.sin(np.power(x, -1))

def sign(x):
    return np.sign(np.sin(8 * x))

# Zmodyfikowana funkcja do interpolacji z użyciem jądra liniowego
def interpolate_function(x, y, kernel, multiplier):
    x_new = np.linspace(x.min(), x.max(), len(x) * multiplier)
    y_new = np.zeros_like(x_new)

    # Zastosowanie konwolucji dla każdego punktu
    for i, xi in enumerate(x_new):
        kernel_scaled = scale_and_shift_kernel(kernel, xi, x)
        y_new[i] = np.sum(y * kernel_scaled) / np.sum(kernel_scaled)

    return x_new, y_new

# Funkcja do skalowania i przesuwania jądra
def scale_and_shift_kernel(kernel, xi, x):
    # Tutaj używam prostego jądra liniowego
    kernel_width = x[1] - x[0]
    distances = np.abs(x - xi)
    kernel_scaled = np.maximum(0, 1 - distances / kernel_width)
    return kernel_scaled

# Parametry
N = 100
multipliers = [2, 4, 10]

# Dane wejściowe
x_original = np.linspace(-np.pi, np.pi, N)
y_original = simplesin(x_original)

# Wykresy
plt.plot(x_original, y_original, label='Original function', color='blue')

# Interpolacja i obliczanie MSE
for multiplier in multipliers:
    # Jądro liniowe
    x_interpolate, y_interpolate = interpolate_function(x_original, y_original, np.ones(3), multiplier)

    mse_original = np.mean((simplesin(x_interpolate) - y_interpolate)**2)
    print(f'MSE for linear kernel with {multiplier} times more points (original function): {mse_original}')

    plt.plot(x_interpolate, y_interpolate, label=f'Linear kernel, {multiplier}x points (original function)', alpha=0.7)

plt.legend()
plt.show()

# Definicja jądra Lanczosa
def lanczos_kernel(a):
    def kernel(x):
        if x == 0:
            return 1
        elif -a < x < a:
            return a * np.sin(np.pi * x) * np.sin(np.pi * x / a) / (np.pi ** 2 * x ** 2)
        else:
            return 0
    return np.vectorize(kernel)

# Parametry i dane
a = 3  # parametr jądra Lanczosa
lanczos = lanczos_kernel(a)  # jądro Lanczosa

# Zmodyfikowany kod z dodanym jądrem Lanczosa
for kernel_name, kernel in [('linear', np.ones(3)), ('Lanczos', lanczos(np.arange(-a, a+1)))]:
    for multiplier in multipliers:

        x_interpolate, y_interpolate = interpolate_function(x_original, y_original, kernel, multiplier)

        mse_original = np.mean((simplesin(x_interpolate) - y_interpolate)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (original function): {mse_original}')

        plt.plot(x_interpolate, y_interpolate, label=f'{kernel_name} kernel, {multiplier}x points (original function)', alpha=0.7)

plt.legend()
plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
