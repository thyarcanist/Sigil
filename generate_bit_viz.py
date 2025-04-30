#!/usr/bin/env python3
"""
Generates a simple black-and-white bit pattern visualization
and its 2D Fourier Transform magnitude spectrum
for a specified entropy source from the IDIA framework.
"""

import os
import sys
import math
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import numpy.fft # Added for FFT
import pywt      # Added for Wavelet Transform

# --- Path Setup (similar to cli.py) ---
# Add project root to sys.path to allow importing idia modules
script_dir = Path(__file__).resolve().parent # src/idia/audit
idia_dir = script_dir.parent                 # src/idia
src_dir = idia_dir.parent                    # src
project_root = src_dir.parent                # Project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
# Also ensure src itself is included for utils etc.
if str(src_dir) not in sys.path:
     sys.path.insert(0, str(src_dir))
# ---------------------------------------

try:
    # Import the necessary components
    from idia.audit.entropy_audit import EntropyAuditor
    # Note: EntropyAuditor internally handles source creation and whitening
except ImportError as e:
    print(f"Error: Failed to import IDIA modules: {e}")
    print("Ensure the script is run from a location where 'src' is accessible")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)

# Helper function to visualize wavelet coefficients
def plot_wavelet_coeffs(coeffs, levels, cmap='gray'):
    """Arranges and plots wavelet coefficients from pywt.wavedec2 into a single image."""
    # coeffs[0] = final LL (approximation)
    # coeffs[i] = (LH, HL, HH) for level i (starting from finest)
    
    # Normalize coefficients for visualization (optional but often helpful)
    # Find max absolute value across all detail coefficients for consistent scaling
    max_abs_val = 0
    for i in range(1, levels + 1):
        for detail_coeff in coeffs[i]: # LH, HL, HH
            max_abs_val = max(max_abs_val, np.max(np.abs(detail_coeff)))
    
    if max_abs_val == 0: max_abs_val = 1 # Avoid division by zero

    # Get dimensions
    final_approx = coeffs[0]
    rows, cols = final_approx.shape
    
    # Create the composite image canvas (same size as original image)
    canvas_rows = rows * (2**levels)
    canvas_cols = cols * (2**levels)
    canvas = np.zeros((canvas_rows, canvas_cols))

    # Place the final approximation (LL) in the top-left corner
    canvas[0:rows, 0:cols] = final_approx / np.max(np.abs(final_approx)) # Scale LL individually

    current_row_offset = 0
    current_col_offset = cols
    for level in range(1, levels + 1):
        lh, hl, hh = coeffs[level]
        level_rows, level_cols = lh.shape # LH, HL, HH have same shape at a level

        # Place LH (Horizontal details) - Top right section for this level
        canvas[current_row_offset : current_row_offset + level_rows, 
               current_col_offset : current_col_offset + level_cols] = lh / max_abs_val
        
        # Place HL (Vertical details) - Bottom left section for this level
        canvas[current_row_offset + level_rows : current_row_offset + 2 * level_rows, 
               current_col_offset - level_cols : current_col_offset] = hl / max_abs_val

        # Place HH (Diagonal details) - Bottom right section for this level
        canvas[current_row_offset + level_rows : current_row_offset + 2 * level_rows, 
               current_col_offset : current_col_offset + level_cols] = hh / max_abs_val

        # Update offsets for the next finer level (details get smaller)
        current_col_offset += level_cols
        # Row offset stays the same for LH, but HL/HH occupy lower part
        # For the *next* level's details, we go back to the top but shift right

    # Display the combined coefficients
    fig, ax = plt.subplots(figsize=(10, 10)) # Larger figure for detail
    im = ax.imshow(canvas, cmap=cmap, vmin=-1, vmax=1) # Scale from -1 to 1
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f'{levels}-Level Wavelet Decomposition')
    # fig.colorbar(im, ax=ax) # Colorbar might be noisy here, often omitted
    return fig

def generate_bit_visualization(source_spec: str, size_bytes: int, output_path: str):
    """Fetches data, converts to bits, reshapes, plots, calculates FFT & Wavelet, 
       plots spectra/coefficients, and saves images."""
    logging.info(f"Generating bit visualization for source: '{source_spec}'")
    logging.info(f"Requesting {size_bytes} bytes.")

    if size_bytes <= 0:
        logging.error("Size must be positive.")
        return

    processed_bytes = None
    source_display_name = source_spec # Default display name

    try:
        # ALWAYS Use EntropyAuditor for data processing
        logging.info("Using EntropyAuditor for data processing.")
        auditor = EntropyAuditor(source_spec=source_spec,
                                 sample_size=size_bytes,
                                 num_samples=1,
                                 chunk_size_samples=1) # Chunk size still not critical here

        # Fetch and process the data using the auditor's pipeline
        _ = auditor.run_all_tests()

        if not hasattr(auditor, 'processed_data') or auditor.processed_data is None:
             raise RuntimeError("EntropyAuditor failed to produce processed data.")

        processed_bytes = auditor.processed_data

        if len(processed_bytes) < size_bytes:
             logging.warning(f"Auditor returned fewer bytes ({len(processed_bytes)}) than requested ({size_bytes}). Visualization may be smaller.")
             if not processed_bytes:
                  raise ValueError("Auditor returned empty data.")

        if processed_bytes is None:
             raise RuntimeError("Failed to obtain processed bytes.")

        # --- Visualization logic (uses processed_bytes) ---
        logging.info(f"Proceeding with visualization using {len(processed_bytes)} processed bytes.")
        bits_array = np.unpackbits(np.frombuffer(processed_bytes, dtype=np.uint8))
        total_bits = len(bits_array)
        logging.info(f"Total bits generated: {total_bits}")

        # Determine dimensions for a square image
        side_length = int(math.sqrt(total_bits))
        if side_length == 0:
             raise ValueError("Not enough bits to form an image.")

        num_pixels = side_length * side_length
        logging.info(f"Creating {side_length}x{side_length} image ({num_pixels} bits).")

        # Trim excess bits and reshape
        bits_to_plot = bits_array[:num_pixels]
        image_array = bits_to_plot.reshape((side_length, side_length))

        # --- Plotting Bit Image ---
        fig_bits, ax_bits = plt.subplots(figsize=(8, 8)) # Adjust size as needed
        ax_bits.imshow(image_array, cmap='binary', interpolation='none')
        ax_bits.set_xticks([])
        ax_bits.set_yticks([])
        ax_bits.set_title(f"Bit Visualization - {source_display_name} ({side_length}x{side_length})", fontsize=14)

        # Ensure output directory exists
        output_dir = os.path.dirname(os.path.abspath(output_path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close(fig_bits) # Close the bit figure
        logging.info(f"Bit Visualization saved to: {output_path}")

        # --- Calculate and Plot FFT Spectrum ---
        logging.info("Calculating 2D FFT Spectrum...")
        fft_result = np.fft.fft2(image_array) # Calculate 2D FFT
        fft_shifted = np.fft.fftshift(fft_result) # Shift zero frequency to center
        # Calculate magnitude spectrum (log scale + 1 to avoid log(0))
        magnitude_spectrum = np.log(np.abs(fft_shifted) + 1)

        # Create a new figure for the FFT
        fig_fft, ax_fft = plt.subplots(figsize=(8, 8))
        # Display the magnitude spectrum (viridis is a common choice)
        im_fft = ax_fft.imshow(magnitude_spectrum, cmap='viridis', interpolation='none')
        ax_fft.set_xticks([])
        ax_fft.set_yticks([])
        ax_fft.set_title(f"FFT Magnitude Spectrum (Log Scale) - {source_display_name}", fontsize=12)
        # Add a colorbar
        fig_fft.colorbar(im_fft, ax=ax_fft)
        
        # Construct FFT output path
        base, ext = os.path.splitext(output_path)
        fft_output_path = f"{base}_fft{ext}"
        
        plt.savefig(fft_output_path, bbox_inches='tight', dpi=150)
        plt.close(fig_fft) # Close the FFT figure
        logging.info(f"FFT Spectrum saved to: {fft_output_path}")

        # --- Calculate and Plot Wavelet Decomposition ---
        try:
            logging.info("Calculating Multiscale Wavelet Decomposition...")
            wavelet_type = 'db4' # Daubechies 4 - a common choice
            decomp_levels = 4    # Number of decomposition levels
            
            # Ensure image dimensions are suitable for chosen levels
            min_dim = min(image_array.shape)
            max_levels = pywt.dwtn_max_level(image_array.shape, wavelet_type)
            actual_levels = min(decomp_levels, max_levels)
            if actual_levels < decomp_levels:
                logging.warning(f"Image size only supports {actual_levels} decomposition levels, not {decomp_levels}. Proceeding with {actual_levels}.")
            if actual_levels == 0:
                raise ValueError("Image too small for wavelet decomposition.")

            # Perform the 2D wavelet decomposition
            # Convert image_array to float for wavelet transform
            coeffs = pywt.wavedec2(image_array.astype(float), wavelet=wavelet_type, level=actual_levels)

            # Plot the coefficients using the helper function
            fig_wavelet = plot_wavelet_coeffs(coeffs, actual_levels, cmap='gray') # Use gray cmap for coefficients
            fig_wavelet.suptitle(f"Wavelet Decomposition ({wavelet_type}, {actual_levels} Levels) - {source_display_name}", fontsize=14)

            # Construct Wavelet output path
            wavelet_output_path = f"{base}_wavelet{ext}"
            
            plt.savefig(wavelet_output_path, bbox_inches='tight', dpi=150)
            plt.close(fig_wavelet) # Close the Wavelet figure
            logging.info(f"Wavelet Decomposition saved to: {wavelet_output_path}")

        except ImportError:
            logging.error("PyWavelets library not found. Skipping Wavelet analysis. Try 'pip install PyWavelets'")
        except Exception as e:
            logging.error(f"Failed during Wavelet analysis: {e}", exc_info=True)

    except Exception as e:
        logging.error(f"Failed to generate visualization, FFT, or Wavelet: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Generate a bit pattern visualization, its FFT spectrum, and Wavelet decomposition for an IDIA entropy source.")
    parser.add_argument("-s", "--source", type=str, default="eris:full",
                        help='Entropy source specification (e.g., "eris", "eris:full", "eris:raw", "system", path/to/file)')
    parser.add_argument("-b", "--size", type=int, default=8192,
                        help="Number of bytes to fetch for the visualization (default: 8192 for 256x256)")
    parser.add_argument("-o", "--output", type=str, required=True,
                        help="Output path for the main PNG image file (FFT and Wavelet images will be saved with suffixes).")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Set the logging level (default: INFO)")

    args = parser.parse_args()

    # Configure Logging
    log_level_int = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level_int, format='%(levelname)s:%(name)s: %(message)s')

    generate_bit_visualization(args.source, args.size, args.output)

if __name__ == "__main__":
    main() 