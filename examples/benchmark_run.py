import os
import sys
import numpy as np
import math # Need math for ceiling calculation
import random # Need standard PRNG
import logging # Added for better status messages
from typing import Optional # <--- Add this import
import argparse # <<< Added for command-line arguments

# --- Path Setup ---
# Adjust path to import from the parent directory (SIGIL)
# This assumes benchmark_run.py is in SIGIL/examples/
# script_dir = os.path.dirname(os.path.abspath(__file__))
# sigil_dir = os.path.dirname(script_dir)
# # Go one level higher to reach the project root containing the .env file
# project_root = os.path.dirname(sigil_dir)
# sys.path.insert(0, sigil_dir)
# # Also add project root for dotenv loading
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)
# # --------------------

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# -------------------------

# Explicitly add project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Import SIGIL components ---
# No change needed here initially, but add API client later
from loaders.entropy_input import load_from_binary_file # Keep for file loading if needed later
from utils.helpers import bytes_to_bit_array
# Import API client functions
from utils.api_client import fetch_eris_full, fetch_eris_raw
from src.idia.entropy_whitening import apply_full_whitening # <<< CORRECT path relative to project root
from analyzers.fft import fft_log_magnitude # Keep FFT
from analyzers.wavelet import wavelet_decompose # Keep Wavelet
from analyzers.visual import generate_bit_visualization # Keep Visual
# Import the stats analyzer functions
from analyzers.stats import frequency_monobit_test, chi_square_byte_distribution_test, runs_test
# Import the report generator
from report.summary import generate_text_summary
# --- Import Local QRNG (conditional) ---
try:
    from src.utils.lattice_framework.qrng import QuantumTrueRandomGenerator
    LOCAL_QRNG_AVAILABLE = True
except ImportError as import_err:
    logging.warning(f"Could not import local QuantumTrueRandomGenerator: {import_err}")
    LOCAL_QRNG_AVAILABLE = False
# --- Import EntropyAuditor ---
try:
    from src.idia.audit.entropy_audit import EntropyAuditor
    AUDITOR_AVAILABLE = True
except ImportError as import_err:
    logging.warning(f"Could not import local EntropyAuditor: {import_err}")
    AUDITOR_AVAILABLE = False
# ------------------------------------

# --- Constants ---
# Set default data size (e.g., ~1.1MB for 3000x3000, or 1MB)
# WIDTH, HEIGHT = 3000, 3000 # Set desired dimensions
# DATA_SIZE_BYTES = math.ceil(WIDTH * HEIGHT / 8) # Calculate bytes needed
WIDTH = 3035 # Match generate_bit_viz for 1152000 bytes
HEIGHT = 3035 # Match generate_bit_viz for 1152000 bytes
DATA_SIZE_BYTES = 1152000 # Match the byte size from generate_bit_viz command
# DATA_SIZE_BYTES = 1125000 # Explicitly calculated for 3000x3000
# DATA_SIZE_BYTES = 1024 * 1024 # 1 MiB - Previous value
# WIDTH = HEIGHT = int(math.sqrt(DATA_SIZE_BYTES * 8)) # Calculate square dimensions for 1MB
OUTPUT_DIR = "./sigil_benchmark_output"
# -----------------

def get_entropy_data(source_name: str, size_bytes: int, use_local_engine: bool = False) -> Optional[bytes]:
    """Fetches or generates entropy data based on the source name.

    Args:
        source_name: The identifier for the entropy source.
        size_bytes: The number of bytes to fetch/generate.
        use_local_engine: If True, use local QRNG for ERIS sources.

    Returns:
        Bytes object containing the entropy data, or None on failure.
    """
    logging.info(f"Getting data for source: {source_name} ({size_bytes} bytes)")

    # <<< Logic for using local engine >>>
    if use_local_engine:
        if source_name == "ERIS:raw":
            if LOCAL_QRNG_AVAILABLE and AUDITOR_AVAILABLE:
                logging.info("--- Using LOCAL EntropyAuditor for ERIS:raw ---")
                try:
                    # Use EntropyAuditor path to mimic generate_bit_viz
                    auditor = EntropyAuditor(source_spec="eris:raw",
                                             sample_size=size_bytes,
                                             num_samples=1, # We only need one large sample
                                             chunk_size_samples=1)
                    _ = auditor.run_all_tests() # Fetch and process data internally
                    if hasattr(auditor, 'processed_data') and auditor.processed_data is not None:
                        if len(auditor.processed_data) >= size_bytes: # Check if enough data was returned
                            logging.info("Successfully retrieved data via EntropyAuditor.")
                            return auditor.processed_data[:size_bytes] # Return requested amount
                        else:
                            logging.error(f"EntropyAuditor returned insufficient data: got {len(auditor.processed_data)}, expected {size_bytes}.")
                            return None
                    else:
                        logging.error("EntropyAuditor failed to produce processed data.")
                        return None
                except Exception as e_auditor:
                    logging.error(f"Error using local EntropyAuditor: {e_auditor}", exc_info=True)
                    return None
            else:
                logging.error("Local QRNG or EntropyAuditor not available. Cannot run ERIS:raw locally.")
                return None
        
        elif source_name == "ERIS:full":
            if LOCAL_QRNG_AVAILABLE:
                logging.info("--- Using LOCAL QRNG + Whitening for ERIS:full ---")
                try:
                    # Fetch RAW data first
                    local_engine = QuantumTrueRandomGenerator()
                    raw_data = local_engine.get_random_bytes(size_bytes)
                    if raw_data is None or len(raw_data) != size_bytes:
                        logging.error(f"Local QRNG generated insufficient raw data for whitening.")
                        return None
                    
                    # Apply whitening
                    logging.info("Applying full whitening to local raw data...")
                    whitened_data_np = apply_full_whitening(raw_data, output_size=size_bytes)
                    data = whitened_data_np.tobytes()
                    
                    if len(data) == size_bytes:
                        logging.info("Full whitening applied successfully locally.")
                        return data
                    else:
                        logging.error(f"Local whitening resulted in unexpected size: {len(data)} bytes")
                        return None
                except Exception as e_local_full:
                    logging.error(f"Error generating local ERIS:full data: {e_local_full}", exc_info=True)
                    return None
            else:
                 logging.error("Local QRNG not available. Cannot run ERIS:full locally.")
                 return None
        else:
            # Handle potential future local sources if needed
            logging.error(f"Local engine mode not implemented for source: {source_name}")
            return None
    # <<< End local engine logic >>>

    # <<< Original API/PRNG logic >>>
    elif source_name == "ERIS:full":
        # Fetch using API client (chunking handled within)
        return fetch_eris_full(size_bytes)
    elif source_name == "ERIS:raw":
        # Fetch using API client (chunking handled within)
        return fetch_eris_raw(size_bytes)
    elif source_name == "PRNG":
        # Use standard library 'random' for basic PRNG
        # Note: random.randbytes requires Python 3.9+
        try:
            return random.randbytes(size_bytes)
        except AttributeError:
            logging.warning("random.randbytes not available (requires Python 3.9+). Using alternative PRNG.")
            # Fallback for older Python versions
            return bytes([random.randint(0, 255) for _ in range(size_bytes)])
    elif source_name == "CSPRNG":
        # Use os.urandom for cryptographically secure PRNG
        return os.urandom(size_bytes)
    else:
        logging.error(f"Unknown data source requested: {source_name}")
        return None


def run_analysis_on_data(data_bytes: bytes, label: str, output_dir: str, width: int, height: int):
    """Runs all available analyses on the provided data bytes and generates outputs."""
    logging.info(f"--- Running Analysis for: {label} ---")
    os.makedirs(output_dir, exist_ok=True)

    # Dictionary to store results
    analysis_results = {
        'label': label,
        'bytes_loaded': len(data_bytes),
        'stats': {},
        'visual_path': None,
        'fft_path': None,
        'wavelet_path': None,
    }

    try:
        # Common setup for image-based analysis
        has_enough_data_for_images = (len(data_bytes) * 8 >= width * height)
        bit_array = None # Initialize bit_array
        if not has_enough_data_for_images:
            logging.warning(f"Warning: Not enough data ({len(data_bytes)} bytes) for a full {width}x{height} analysis. Skipping image-based analyses.")
        else:
            try:
                bit_array = bytes_to_bit_array(data_bytes, width, height)
            except MemoryError:
                logging.error(f"MemoryError converting bytes to bit_array for {label}. Skipping image-based analyses.")
                bit_array = None # Ensure bit_array is None if conversion fails
            except ValueError as ve_bitarray:
                logging.error(f"ValueError converting bytes to bit_array for {label}: {ve_bitarray}. Skipping image-based analyses.")
                bit_array = None

        # Run Visual Analysis
        if bit_array is not None:
            try:
                logging.info("Generating bit visualization...")
                # Use consistent file naming, width/height are already defined
                safe_label = label.replace(':', '_').replace('/', '_')
                viz_path = os.path.join(output_dir, f"{safe_label}_bit_visualization_{width}x{height}.png")
                from PIL import Image
                img = Image.fromarray((bit_array * 255).astype(np.uint8), 'L')
                # --- Add title matching generate_bit_viz --- #
                import matplotlib.pyplot as plt
                fig_bits, ax_bits = plt.subplots(figsize=(8, 8))
                ax_bits.imshow(img, cmap='binary', interpolation='none')
                ax_bits.set_xticks([])
                ax_bits.set_yticks([])
                ax_bits.set_title(f"Bit Visualization - {label} ({width}x{height})", fontsize=14) # Match generate_bit_viz title
                plt.savefig(viz_path, bbox_inches='tight', dpi=150) # Save figure, not raw image
                plt.close(fig_bits) # Close the figure
                # img.save(viz_path) # Original save method
                # --- End Title/Save Change ---
                analysis_results['visual_path'] = viz_path # Record path only on success
                logging.info(f"Bit visualization saved to: {viz_path}")
            except MemoryError:
                logging.error(f"MemoryError generating/saving visual analysis for {label}.")
            except Exception as e_visual:
                logging.error(f"Error during visual analysis for {label}: {e_visual}")
        # else: # No need for else if bit_array is None, handled by initial check
            # logging.info("Skipping visual analysis due to insufficient data or bit_array error.")

        # Run FFT Analysis
        if bit_array is not None:
            try:
                logging.info("Performing FFT analysis...")
                fft_result = fft_log_magnitude(bit_array) # Potential MemoryError here
                fft_plot_path = os.path.join(output_dir, f"{safe_label}_fft_spectrum_{width}x{height}.png") # Sanitize label
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 8))
                plt.imshow(fft_result, cmap="viridis")
                plt.colorbar()
                # plt.title(f"FFT Magnitude Spectrum - {label}") # Original title
                plt.title(f"FFT Magnitude Spectrum (Log Scale) - {label}", fontsize=12) # Match generate_bit_viz title
                plt.savefig(fft_plot_path)
                plt.close()
                analysis_results['fft_path'] = fft_plot_path # Record path only on success
                logging.info(f"FFT spectrum saved to: {fft_plot_path}")
            except MemoryError:
                logging.error(f"MemoryError during FFT analysis/plotting for {label}.")
            except Exception as e_fft:
                logging.error(f"Error during FFT analysis for {label}: {e_fft}")
        # else:
            # logging.info("Skipping FFT analysis due to insufficient data or bit_array error.")

        # Run Wavelet Analysis
        if bit_array is not None:
            try:
                logging.info("Performing Wavelet analysis...")
                wavelet_plot_path = os.path.join(output_dir, f"{safe_label}_wavelet_decomp_{width}x{height}.png") # Use safe_label
                import matplotlib.pyplot as plt
                import pywt # <<< Import pywt here

                # --- Replicate generate_bit_viz Wavelet logic --- #
                wavelet_type = 'db4' # Match generate_bit_viz
                decomp_levels = 4    # Match generate_bit_viz
                
                # Ensure image dimensions are suitable for chosen levels (borrowed from generate_bit_viz)
                min_dim = min(bit_array.shape)
                max_levels = pywt.dwtn_max_level(bit_array.shape, wavelet_type)
                actual_levels = min(decomp_levels, max_levels)
                if actual_levels < decomp_levels:
                    logging.warning(f"Image size only supports {actual_levels} decomposition levels, not {decomp_levels}. Proceeding with {actual_levels}.")
                if actual_levels == 0:
                    raise ValueError("Image too small for wavelet decomposition.")

                # Perform the 2D wavelet decomposition directly
                coeffs = pywt.wavedec2(bit_array.astype(float), wavelet=wavelet_type, level=actual_levels)

                # --- Plotting logic from plot_wavelet_coeffs --- #
                fig_wavelet, ax_wavelet = plt.subplots(figsize=(10, 10)) # Match size

                # Normalize coefficients (borrowed from generate_bit_viz)
                max_abs_val = 0
                for i in range(1, actual_levels + 1):
                    for detail_coeff in coeffs[i]: # LH, HL, HH
                        current_max = np.max(np.abs(detail_coeff))
                        if not np.isnan(current_max) and not np.isinf(current_max):
                             max_abs_val = max(max_abs_val, current_max)
                if max_abs_val == 0: max_abs_val = 1 # Avoid division by zero

                # Get dimensions (borrowed from generate_bit_viz)
                final_approx = coeffs[0]
                approx_max = np.max(np.abs(final_approx))
                if approx_max == 0: approx_max = 1 # Avoid division by zero if approx is all zero
                
                rows, cols = final_approx.shape

                # Create canvas (borrowed from generate_bit_viz)
                # Ensure canvas matches potentially smaller rows/cols if levels reduced
                if rows > 0 and cols > 0:
                    canvas_rows = rows * (2**actual_levels)
                    canvas_cols = cols * (2**actual_levels)
                    # Check if canvas size matches bit_array dimensions, adjust if necessary
                    if canvas_rows > height or canvas_cols > width:
                         logging.warning(f"Wavelet canvas size ({canvas_rows}x{canvas_cols}) exceeds image size ({height}x{width}) due to level reduction. Clipping canvas.")
                         canvas_rows = height
                         canvas_cols = width
                         # We might need to adjust coefficient placement logic if clipped - simpler to ensure image size allows levels

                    canvas = np.zeros((canvas_rows, canvas_cols))

                    # Place approximation (borrowed from generate_bit_viz)
                    canvas[0:rows, 0:cols] = final_approx / approx_max # Scale LL individually

                    current_row_offset = 0
                    current_col_offset = cols
                    for level in range(1, actual_levels + 1):
                        lh, hl, hh = coeffs[level]
                        level_rows, level_cols = lh.shape

                        # Check bounds before placing coefficients
                        if current_row_offset + 2 * level_rows <= canvas_rows and \
                           current_col_offset + level_cols <= canvas_cols:
                           
                            # Normalize detail coefficients before placing
                            lh_norm = lh / max_abs_val
                            hl_norm = hl / max_abs_val
                            hh_norm = hh / max_abs_val

                            # Place LH
                            canvas[current_row_offset : current_row_offset + level_rows, 
                                   current_col_offset : current_col_offset + level_cols] = lh_norm
                            # Place HL
                            canvas[current_row_offset + level_rows : current_row_offset + 2 * level_rows, 
                                   current_col_offset - level_cols : current_col_offset] = hl_norm
                            # Place HH
                            canvas[current_row_offset + level_rows : current_row_offset + 2 * level_rows, 
                                   current_col_offset : current_col_offset + level_cols] = hh_norm

                            # Update offsets (borrowed from generate_bit_viz)
                            current_col_offset += level_cols
                        else:
                             logging.warning(f"Skipping coefficient placement for level {level} due to canvas boundary.")
                             break # Stop placing if out of bounds
                    
                    # Display the canvas (borrowed from generate_bit_viz)
                    im = ax_wavelet.imshow(canvas, cmap='gray', vmin=-1, vmax=1)
                    ax_wavelet.set_xticks([])
                    ax_wavelet.set_yticks([])
                    # ax_wavelet.set_title(f'{actual_levels}-Level Wavelet Decomposition') # Original helper title
                    fig_wavelet.suptitle(f"Wavelet Decomposition ({wavelet_type}, {actual_levels} Levels) - {label}", fontsize=14) # Match generate_bit_viz title

                else:
                     logging.error(f"Wavelet approximation dimensions are zero for {label}. Cannot create canvas.")
                     ax_wavelet.text(0.5, 0.5, 'Wavelet analysis failed (zero dimensions)', ha='center', va='center')

                # --- End Plotting Logic --- #
                
                # plt.figure(figsize=(8, 8))
                # plt.imshow(wavelet_result, cmap='gray')
                # plt.title(f"Wavelet Decomposition ({wavelet_type}, {decomp_levels} Levels) - {label}", fontsize=14)
                plt.savefig(wavelet_plot_path, bbox_inches='tight', dpi=150) # Save the constructed figure
                plt.close(fig_wavelet) # Close the wavelet figure
                # --- End Wavelet Logic Replication ---

                analysis_results['wavelet_path'] = wavelet_plot_path # Record path only on success
                logging.info(f"Wavelet decomposition saved to: {wavelet_plot_path}")
            except MemoryError:
                logging.error(f"MemoryError during Wavelet analysis/plotting for {label}.")
            except ImportError:
                 logging.error("PyWavelets library not found. Skipping Wavelet analysis. Try 'pip install PyWavelets'")
            except Exception as e_wavelet:
                 logging.error(f"Error during Wavelet analysis for {label}: {e_wavelet}")
        # else:
             # logging.info("Skipping Wavelet analysis due to insufficient data or bit_array error.")

        # Run Stats Analysis (Assuming less memory intensive)
        logging.info("Performing Frequency (Monobit) Test...")
        freq_test_result = frequency_monobit_test(data_bytes)
        analysis_results['stats']['frequency_monobit'] = freq_test_result
        logging.info("Performing Chi-Square Byte Distribution Test...")
        chisq_test_result = chi_square_byte_distribution_test(data_bytes)
        analysis_results['stats']['chi_square_byte'] = chisq_test_result
        logging.info("Performing Runs Test...")
        runs_test_result = runs_test(data_bytes)
        analysis_results['stats']['runs'] = runs_test_result

        # TODO: Add calls to other Stats Analyzers

    # --- Keep the outer broad exception handler for unexpected issues --- #
    except MemoryError:
        # This catches MemoryError if it happens *outside* the specific analysis blocks
        # (e.g., maybe during initial setup within the main try)
        logging.error(f"MEMORY ERROR occurred processing {label}. Cannot proceed with analysis.")
        # Reset image paths if error happened before summary generation
        analysis_results['visual_path'] = None
        analysis_results['fft_path'] = None
        analysis_results['wavelet_path'] = None
    except Exception as e:
        logging.error(f"An unexpected general error occurred during analysis for {label}: {e}", exc_info=True)
        # Reset image paths if error happened before summary generation
        analysis_results['visual_path'] = None
        analysis_results['fft_path'] = None
        analysis_results['wavelet_path'] = None
    # ------------------------------------------------------------------- #

    # Generate Report (always attempt to generate, using None for paths if saving failed)
    logging.info("Generating analysis summary...")
    # Sanitize label for filename
    safe_label = label.replace(":", "_").replace("/", "_")
    summary_path = os.path.join(output_dir, f"{safe_label}_analysis_summary.txt")
    # Pass the potentially updated analysis_results (with None for paths if errors occurred)
    generate_text_summary(analysis_results, summary_path)

    logging.info(f"--- Analysis complete for: {label} ---\n")


if __name__ == "__main__":
    # <<< Setup Argument Parser >>>
    parser = argparse.ArgumentParser(description="Run SIGIL benchmark analyses on entropy sources.")
    parser.add_argument(
        '--local-engine',
        action='store_true',
        help='Use local QuantumTrueRandomGenerator for ERIS sources instead of API calls.'
    )
    args = parser.parse_args()
    # <<< End Argument Parser Setup >>>

    logging.info("Starting SIGIL Benchmark Run...")
    if args.local_engine:
        logging.info("*** LOCAL ENGINE MODE ACTIVATED ***")
        if not LOCAL_QRNG_AVAILABLE:
            logging.error("Local engine mode requested, but the required local QRNG module could not be imported. Exiting.")
            sys.exit(1)

    # Ensure matplotlib is imported if needed by analysis functions
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.warning("Matplotlib not found. Plot generation will fail.")

    # --- Configuration ---\
    data_sources_to_run = ["ERIS:raw", "ERIS:full", "PRNG", "CSPRNG"]
    # ---------------------\

    # Run analysis on each source
    for source_label in data_sources_to_run:
        # Get data for the current source
        entropy_data = get_entropy_data(source_label, DATA_SIZE_BYTES, use_local_engine=args.local_engine)

        if entropy_data:
            if len(entropy_data) == DATA_SIZE_BYTES:
                # Run the analysis pipeline
                run_analysis_on_data(entropy_data, source_label, OUTPUT_DIR, WIDTH, HEIGHT)
            else:
                logging.warning(f"Received insufficient data for {source_label}. Expected {DATA_SIZE_BYTES}, got {len(entropy_data)}. Skipping analysis.")
        else:
            logging.error(f"Failed to get data for {source_label}. Skipping analysis.")

    logging.info("SIGIL Benchmark Run Finished.")
    logging.info(f"Check the '{OUTPUT_DIR}' folder for results.")
    # Remove the old dummy file generation and loop
    # ... (Previous dummy file logic removed) ...
