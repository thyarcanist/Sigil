# fft.py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2

def fft_log_magnitude(bit_array_2d):
    fft_result = np.abs(fft2(bit_array_2d))
    log_mag = np.log1p(fft_result)
    return log_mag

def plot_fft(log_mag, title="FFT Magnitude Spectrum"):
    plt.imshow(log_mag, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.show()
