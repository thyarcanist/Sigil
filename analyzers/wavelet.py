# wavelet.py (Modified to match generate_bit_viz.py)
import numpy as np
import pywt
import matplotlib.pyplot as plt
import logging # Added for warnings

# Set up a basic logger if not configured elsewhere
# logging.basicConfig(level=logging.INFO)

def wavelet_decompose(data_2d, level=4, wavelet='db4'):
    """
    Performs 2D wavelet decomposition using pywt.wavedec2.
    Returns the coefficients as a list and the actual levels used.
    Matches the decomposition step in generate_bit_viz.py.

    Args:
        data_2d (np.ndarray): The 2D input image array.
        level (int): The desired decomposition level.
        wavelet (str): The wavelet type (e.g., 'db4').

    Returns:
        tuple: (list_of_coeffs, actual_levels_used)
               Returns (None, 0) if decomposition fails.
    """
    try:
        # Ensure input is float
        data_float = data_2d.astype(float)

        # Ensure image dimensions are suitable for chosen levels
        max_levels = pywt.dwtn_max_level(data_float.shape, wavelet)
        actual_levels = min(level, max_levels)

        if actual_levels < level:
            logging.warning(f"Image size only supports {actual_levels} decomposition levels, not {level}. Proceeding with {actual_levels}.")
        if actual_levels == 0:
            logging.error("Image too small for any wavelet decomposition.")
            return None, 0

        # Perform the 2D wavelet decomposition
        coeffs = pywt.wavedec2(data_float, wavelet=wavelet, level=actual_levels)
        return coeffs, actual_levels

    except Exception as e:
        logging.error(f"Wavelet decomposition failed: {e}", exc_info=True)
        return None, 0


def plot_wavelet_coeffs(coeffs, levels, cmap='gray', title="Wavelet Decomposition"):
    """
    Arranges and plots wavelet coefficients from pywt.wavedec2 list
    into a single image, mimicking generate_bit_viz.py's visualization.

    Args:
        coeffs (list): The coefficients list from wavelet_decompose.
        levels (int): The number of decomposition levels used.
        cmap (str): Colormap for the plot.
        title (str): Title for the plot.

    Returns:
        tuple: (figure, axes) objects from matplotlib. Returns (None, None) on error.
    """
    if coeffs is None or levels <= 0:
        logging.error("Invalid coefficients or levels for plotting.")
        return None, None

    try:
        # Normalize detail coefficients for visualization
        max_abs_val = 0
        for i in range(1, levels + 1):
             # Ensure coeffs[i] exists and is iterable (tuple of LH, HL, HH)
             if i < len(coeffs) and isinstance(coeffs[i], (list, tuple)) and len(coeffs[i]) == 3:
                for detail_coeff in coeffs[i]: # LH, HL, HH
                    if detail_coeff is not None:
                        max_abs_val = max(max_abs_val, np.max(np.abs(detail_coeff)))
             else:
                 logging.warning(f"Coefficient structure unexpected at level {i}. Skipping level for normalization.")


        if max_abs_val == 0: max_abs_val = 1 # Avoid division by zero

        # Get dimensions from the final approximation (LL)
        final_approx = coeffs[0]
        if final_approx is None:
            logging.error("Final approximation coefficient (LL) is missing.")
            return None, None
        rows, cols = final_approx.shape

        # Create the composite image canvas (intended final size)
        # Note: generate_bit_viz assumes original image size, let's calculate based on LL
        canvas_rows = rows * (2**levels)
        canvas_cols = cols * (2**levels)
        canvas = np.zeros((canvas_rows, canvas_cols))

        # Place the final approximation (LL) - Scale individually
        ll_max_abs = np.max(np.abs(final_approx))
        if ll_max_abs > 0:
             canvas[0:rows, 0:cols] = final_approx / ll_max_abs
        else:
             canvas[0:rows, 0:cols] = final_approx # Already zero or constant

        current_row_offset = 0
        current_col_offset = cols
        for level in range(1, levels + 1):
            # Check if level exists and has the expected structure
            if level >= len(coeffs) or not isinstance(coeffs[level], (list, tuple)) or len(coeffs[level]) != 3:
                 logging.warning(f"Skipping plotting for level {level} due to unexpected coefficient structure.")
                 # Update offsets roughly to avoid overlap, though layout might be wrong
                 level_rows = rows * (2**(levels - level))
                 level_cols = cols * (2**(levels - level))
                 current_col_offset += level_cols
                 continue

            lh, hl, hh = coeffs[level]
            if lh is None or hl is None or hh is None:
                 logging.warning(f"Skipping plotting for level {level} due to missing LH, HL, or HH coefficients.")
                 level_rows = rows * (2**(levels - level))
                 level_cols = cols * (2**(levels - level))
                 current_col_offset += level_cols
                 continue

            level_rows, level_cols = lh.shape # LH, HL, HH have same shape

            # Calculate placement boundaries carefully
            row_start_lh = current_row_offset
            row_end_lh = current_row_offset + level_rows
            col_start_lh = current_col_offset
            col_end_lh = current_col_offset + level_cols

            row_start_hl = current_row_offset + level_rows
            row_end_hl = current_row_offset + 2 * level_rows
            col_start_hl = current_col_offset - level_cols
            col_end_hl = current_col_offset

            row_start_hh = current_row_offset + level_rows
            row_end_hh = current_row_offset + 2 * level_rows
            col_start_hh = current_col_offset
            col_end_hh = current_col_offset + level_cols

            # Check if indices are within canvas bounds
            if row_end_lh > canvas_rows or col_end_lh > canvas_cols or \
               row_end_hl > canvas_rows or col_start_hl < 0 or \
               row_end_hh > canvas_rows or col_end_hh > canvas_cols:
                logging.warning(f"Calculated indices for level {level} are out of canvas bounds. Skipping placement.")
                # Update offset roughly
                current_col_offset += level_cols
                continue

            # Place LH (normalize using max_abs_val)
            canvas[row_start_lh : row_end_lh, col_start_lh : col_end_lh] = lh / max_abs_val

            # Place HL
            canvas[row_start_hl : row_end_hl, col_start_hl : col_end_hl] = hl / max_abs_val

            # Place HH
            canvas[row_start_hh : row_end_hh, col_start_hh : col_end_hh] = hh / max_abs_val

            # Update column offset for the next level
            current_col_offset += level_cols
            # Row offset for the next iteration's details (LH band) resets to the top band of the current level block
            # The calculation in generate_bit_viz.py seemed to imply row offset didn't change,
            # but the visualization requires careful placement. Let's stick to the example's apparent layout.

        # Display the combined coefficients
        fig, ax = plt.subplots(figsize=(10, 10))
        im = ax.imshow(canvas, cmap=cmap, vmin=-1, vmax=1) # Scale from -1 to 1
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title} ({levels}-Level)", fontsize=14) # Match script's title style
        # fig.colorbar(im, ax=ax) # Optional colorbar

        return fig, ax

    except Exception as e:
        logging.error(f"Wavelet plotting failed: {e}", exc_info=True)
        return None, None


# --- Example Usage (similar to how it might be used) ---
# if __name__ == '__main__':
#     # Create a dummy 2D array (e.g., like the bit image)
#     dummy_side = 256
#     dummy_image = np.random.rand(dummy_side, dummy_side) # Use rand for float data
#
#     # Decompose the image
#     desired_levels = 4
#     coeffs, actual_levels = wavelet_decompose(dummy_image, level=desired_levels, wavelet='db4')
#
#     # Plot the result if decomposition was successful
#     if coeffs and actual_levels > 0:
#         fig, ax = plot_wavelet_coeffs(coeffs, actual_levels, title="Example Wavelet Decomposition")
#         if fig:
#             # If you want to show it immediately:
#             plt.show()
#
#             # Or save it:
#             # fig.savefig("example_wavelet.png", bbox_inches='tight', dpi=150)
#             # plt.close(fig) # Close the figure after saving if not showing
#     else:
#         print("Could not perform or plot wavelet decomposition.")
