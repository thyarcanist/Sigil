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
# ------------------------------------

# --- Constants ---
# Set default data size (e.g., ~1.1MB for 3000x3000, or 1MB)
# WIDTH, HEIGHT = 3000, 3000
# DATA_SIZE_BYTES = math.ceil(WIDTH * HEIGHT / 8)
DATA_SIZE_BYTES = 1024 * 1024 # 1 MiB
WIDTH = HEIGHT = int(math.sqrt(DATA_SIZE_BYTES * 8)) # Calculate square dimensions for 1MB
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
    if use_local_engine and source_name in ["ERIS:raw", "ERIS:full"]:
        if LOCAL_QRNG_AVAILABLE:
            logging.info(f"--- Using LOCAL QuantumTrueRandomGenerator for {source_name} --- ")
            try:
                local_engine = QuantumTrueRandomGenerator() # Instantiate local engine
                # --- Get RAW bytes first ---
                raw_data = local_engine.get_random_bytes(size_bytes)
                if raw_data is None or len(raw_data) != size_bytes:
                    logging.error(f"Local QRNG generated insufficient or None data ({len(raw_data) if raw_data else 'None'} bytes), expected {size_bytes}.")
                    return None

                # --- Apply Whitening ONLY for ERIS:full ---
                if source_name == "ERIS:full":
                    logging.info("Applying full whitening to local raw data...")
                    try:
                        # Pass raw_data (bytes) and expected output size
                        # Use the direct import again
                        whitened_data_np = apply_full_whitening(raw_data, output_size=size_bytes)
                        data = whitened_data_np.tobytes() # Convert result to bytes

                        if len(data) == size_bytes:
                             logging.info("Full whitening applied successfully locally.")
                             return data
                        else:
                             logging.error(f"Local whitening resulted in unexpected size: {len(data)} bytes, expected {size_bytes}.")
                             return None
                    except Exception as e_whitening:
                        logging.error(f"Error applying full whitening locally: {e_whitening}", exc_info=True)
                        return None
                else: # ERIS:raw
                    logging.info("Returning raw data for ERIS:raw (local engine).")
                    return raw_data # Return raw bytes directly

            except Exception as e_local_qrng:
                logging.error(f"Error using local QuantumTrueRandomGenerator: {e_local_qrng}", exc_info=True)
                return None
        else:
            logging.error(f"Local engine requested for {source_name}, but local QRNG is not available (Import failed). Skipping.")
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
                viz_path = os.path.join(output_dir, f"{label.replace(':', '_')}_bit_visualization_{width}x{height}.png") # Sanitize label
                from PIL import Image
                img = Image.fromarray((bit_array * 255).astype(np.uint8), 'L')
                img.save(viz_path)
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
                fft_plot_path = os.path.join(output_dir, f"{label.replace(':', '_')}_fft_spectrum_{width}x{height}.png") # Sanitize label
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 8))
                plt.imshow(fft_result, cmap="viridis")
                plt.colorbar()
                plt.title(f"FFT Magnitude Spectrum - {label}")
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
                wavelet_result = wavelet_decompose(bit_array) # Potential MemoryError here
                wavelet_plot_path = os.path.join(output_dir, f"{label.replace(':', '_')}_wavelet_decomp_{width}x{height}.png") # Sanitize label
                import matplotlib.pyplot as plt
                plt.figure(figsize=(8, 8))
                plt.imshow(wavelet_result, cmap='gray')
                plt.title(f"Wavelet Decomposition - {label}")
                plt.savefig(wavelet_plot_path)
                plt.close()
                analysis_results['wavelet_path'] = wavelet_plot_path # Record path only on success
                logging.info(f"Wavelet decomposition saved to: {wavelet_plot_path}")
            except MemoryError:
                logging.error(f"MemoryError during Wavelet analysis/plotting for {label}.")
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
