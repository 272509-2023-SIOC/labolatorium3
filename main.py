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

# Definicja jądra sinc
def sinc_kernel(a):
    def kernel(x):
        if x == 0:
            return 1
        else:
            return np.sinc(x / a)
    return np.vectorize(kernel)

# Parametry
N = 100
multipliers = [2, 4, 10]
a = 3  # parametr dla jąder Lanczosa i sinc

# Dane wejściowe
x_original = np.linspace(-np.pi, np.pi, N)
y_original = simplesin(x_original)

# Wykresy dla różnych jąder
kernels = [('linear', np.ones(3)), ('Lanczos', lanczos_kernel(a)(np.arange(-a, a+1))), ('sinc', sinc_kernel(a)(np.linspace(-a, a, 2*a+1)))]
for kernel_name, kernel in kernels:
    for multiplier in multipliers:
        x_interpolate, y_interpolate = interpolate_function(x_original, y_original, kernel, multiplier)
        mse_original = np.mean((simplesin(x_interpolate) - y_interpolate)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (original function): {mse_original}')
        plt.plot(x_interpolate, y_interpolate, label=f'{kernel_name} kernel, {multiplier}x points (original function)', alpha=0.7)

plt.legend()
plt.show()

# Dane wejściowe
x_original = np.linspace(-np.pi, np.pi, N, endpoint=False)  # Wykluczenie punktu x=0 dla funkcji inverted_sin

# Wykresy dla różnych jąder i funkcji
plt.figure(figsize=(15, 10))

# Funkcja inverted_sin
plt.subplot(2, 1, 1)
y_original_inverted = inverted_sin(x_original)
plt.plot(x_original, y_original_inverted, label='Original inverted_sin function', color='blue', alpha=0.7)

for kernel_name, kernel in kernels:
    for multiplier in multipliers:
        x_interpolate, y_interpolate = interpolate_function(x_original, y_original_inverted, kernel, multiplier)
        mse_inverted = np.mean((inverted_sin(x_interpolate) - y_interpolate)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (inverted_sin function): {mse_inverted}')
        plt.plot(x_interpolate, y_interpolate, label=f'{kernel_name} kernel, {multiplier}x points (inverted_sin function)', alpha=0.7)

plt.legend()
plt.title("Interpolation of Inverted Sin Function")

# Funkcja sign
plt.subplot(2, 1, 2)
y_original_sign = sign(x_original)
plt.plot(x_original, y_original_sign, label='Original sign function', color='blue', alpha=0.7)

for kernel_name, kernel in kernels:
    for multiplier in multipliers:
        x_interpolate, y_interpolate = interpolate_function(x_original, y_original_sign, kernel, multiplier)
        mse_sign = np.mean((sign(x_interpolate) - y_interpolate)**2)
        print(f'MSE for {kernel_name} kernel with {multiplier} times more points (sign function): {mse_sign}')
        plt.plot(x_interpolate, y_interpolate, label=f'{kernel_name} kernel, {multiplier}x points (sign function)', alpha=0.7)

plt.legend()
plt.title("Interpolation of Sign Function")

plt.tight_layout()
plt.show()
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
