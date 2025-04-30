# fft.py
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def fft_log_magnitude(bit_array_2d):
    fft_result = fft2(bit_array_2d)
    fft_shifted = fftshift(fft_result)
    log_mag = np.log1p(np.abs(fft_shifted))
    return log_mag

def plot_fft(log_mag, title="FFT Magnitude Spectrum"):
    plt.imshow(log_mag, cmap="viridis")
    plt.colorbar()
    plt.title(title)
    plt.show()
