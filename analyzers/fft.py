# fft.py (Modified to match generate_bit_viz.py)
import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, fftshift

def fft_log_magnitude(bit_array_2d):
    """
    Calculates the 2D FFT and returns the log magnitude spectrum,
    shifted so the DC component is at the center.
    Matches the calculation in generate_bit_viz.py.
    """
    # Ensure input is suitable for FFT (e.g., float)
    # generate_bit_viz.py implicitly casts to float if needed by fft2,
    # but explicit casting might be safer depending on input type.
    # Assuming bit_array_2d is already numeric (0s and 1s).
    fft_result = fft2(bit_array_2d)
    fft_shifted = fftshift(fft_result)
    # Calculate log magnitude (adding 1 to avoid log(0))
    magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)
    return magnitude_spectrum

def plot_fft(magnitude_spectrum, title="FFT Magnitude Spectrum (Log Scale)"):
    """
    Plots the pre-calculated FFT log magnitude spectrum.
    Matches the plotting style in generate_bit_viz.py.
    Returns the figure and axes objects.
    """
    fig, ax = plt.subplots(figsize=(8, 8)) # Use a default size like in the script
    im = ax.imshow(magnitude_spectrum, cmap="viridis", interpolation='none')
    fig.colorbar(im, ax=ax)
    ax.set_title(title, fontsize=12) # Match font size setting style
    # Remove ticks for cleaner image, like in the script
    ax.set_xticks([])
    ax.set_yticks([])
    # Removed plt.show() - function now returns fig, ax for external control
    return fig, ax

# --- Example Usage (similar to how it might be used) ---
# if __name__ == '__main__':
#     # Create a dummy 2D bit array
#     dummy_side = 256
#     dummy_image = np.random.randint(0, 2, size=(dummy_side, dummy_side))
#
#     # Calculate the FFT log magnitude
#     fft_mag = fft_log_magnitude(dummy_image)
#
#     # Plot the result
#     fig, ax = plot_fft(fft_mag, title="Example FFT Spectrum")
#
#     # If you want to show it immediately:
#     plt.show()
#
#     # Or save it:
#     # fig.savefig("example_fft.png", bbox_inches='tight', dpi=150)
#     # plt.close(fig) # Close the figure after saving if not showing