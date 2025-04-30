# wavelet.py
import pywt
import matplotlib.pyplot as plt

def wavelet_decompose(data, level=4, wavelet='db4'):
    coeffs = pywt.wavedec2(data, wavelet=wavelet, level=level)
    return pywt.coeffs_to_array(coeffs)[0]

def plot_wavelet(decomp, title="Wavelet Decomposition"):
    plt.imshow(decomp, cmap='gray')
    plt.title(title)
    plt.show()
