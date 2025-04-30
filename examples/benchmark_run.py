import os
import sys
import numpy as np
import math # Need math for ceiling calculation
import random # Need standard PRNG
import logging # Added for better status messages

# --- Path Setup ---
# Adjust path to import from the parent directory (SIGIL)
# This assumes benchmark_run.py is in SIGIL/examples/
script_dir = os.path.dirname(os.path.abspath(__file__))
sigil_dir = os.path.dirname(script_dir)
# Go one level higher to reach the project root containing the .env file
project_root = os.path.dirname(sigil_dir)
sys.path.insert(0, sigil_dir)
# Also add project root for dotenv loading
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --------------------

# --- Configure Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# -------------------------

# --- Import SIGIL components ---
# No change needed here initially, but add API client later
from loaders.entropy_input import load_from_binary_file # Keep for file loading if needed later
from utils.helpers import bytes_to_bit_array
# Import API client functions
from utils.api_client import fetch_eris_full, fetch_eris_raw
from analyzers.fft import fft_log_magnitude # Keep FFT
from analyzers.wavelet import wavelet_decompose # Keep Wavelet
from analyzers.visual import generate_bit_visualization # Keep Visual
# Import the stats analyzer functions
from analyzers.stats import frequency_monobit_test, chi_square_byte_distribution_test, runs_test
# Import the report generator
from report.summary import generate_text_summary
# ---------------------------

# --- Constants ---
# Set default data size (e.g., ~1.1MB for 3000x3000, or 1MB)
# WIDTH, HEIGHT = 3000, 3000
# DATA_SIZE_BYTES = math.ceil(WIDTH * HEIGHT / 8)
DATA_SIZE_BYTES = 1024 * 1024 # 1 MiB
WIDTH = HEIGHT = int(math.sqrt(DATA_SIZE_BYTES * 8)) # Calculate square dimensions for 1MB
OUTPUT_DIR = "./sigil_benchmark_output"
# -----------------

def get_entropy_data(source_name: str, size_bytes: int) -> Optional[bytes]:
    """Fetches or generates entropy data based on the source name."""
    logging.info(f"Getting data for source: {source_name} ({size_bytes} bytes)")
    if source_name == "ERIS:full":
        return fetch_eris_full(size_bytes)
    elif source_name == "ERIS:raw":
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

        if not has_enough_data_for_images:
            logging.warning(f"Warning: Not enough data ({len(data_bytes)} bytes) for a full {width}x{height} analysis. Skipping image-based analyses.")
            bit_array = None
        else:
             bit_array = bytes_to_bit_array(data_bytes, width, height)

        # Run Visual Analysis
        if bit_array is not None:
            logging.info("Generating bit visualization...")
            viz_path = os.path.join(output_dir, f"{label}_bit_visualization_{width}x{height}.png")
            # Pass data_bytes directly to the updated visualizer if needed, or keep bit_array
            # Assuming visualizer needs the raw bytes and dimensions
            # generate_bit_visualization(data_bytes, width, height, viz_path) # If visualizer takes bytes
            # If visualizer takes bit array directly:
            # Recreate image from bit_array for generate_bit_visualization
            from PIL import Image
            img = Image.fromarray((bit_array * 255).astype(np.uint8), 'L')
            img.save(viz_path)
            analysis_results['visual_path'] = viz_path
            logging.info(f"Bit visualization saved to: {viz_path}")
        else:
            logging.info("Skipping visual analysis due to insufficient data.")

        # Run FFT Analysis
        if bit_array is not None:
            logging.info("Performing FFT analysis...")
            fft_result = fft_log_magnitude(bit_array)
            fft_plot_path = os.path.join(output_dir, f"{label}_fft_spectrum_{width}x{height}.png")
            import matplotlib.pyplot as plt # Keep import local? Or move to top? Move to top is cleaner.
            plt.figure(figsize=(8, 8)) # Consistent figure size
            plt.imshow(fft_result, cmap="viridis")
            plt.colorbar()
            plt.title(f"FFT Magnitude Spectrum - {label}")
            plt.savefig(fft_plot_path)
            plt.close() # Close the figure to free memory
            logging.info(f"FFT spectrum saved to: {fft_plot_path}")
            analysis_results['fft_path'] = fft_plot_path
        else:
            logging.info("Skipping FFT analysis due to insufficient data.")

        # Run Wavelet Analysis
        if bit_array is not None:
            logging.info("Performing Wavelet analysis...")
            wavelet_result = wavelet_decompose(bit_array)
            wavelet_plot_path = os.path.join(output_dir, f"{label}_wavelet_decomp_{width}x{height}.png")
            plt.figure(figsize=(8, 8)) # Consistent figure size
            plt.imshow(wavelet_result, cmap='gray')
            plt.title(f"Wavelet Decomposition - {label}")
            plt.savefig(wavelet_plot_path)
            plt.close()
            logging.info(f"Wavelet decomposition saved to: {wavelet_plot_path}")
            analysis_results['wavelet_path'] = wavelet_plot_path
        else:
             logging.info("Skipping Wavelet analysis due to insufficient data.")

        # Run Stats Analysis
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

        # Generate Report
        logging.info("Generating analysis summary...")
        # Sanitize label for filename
        safe_label = label.replace(":", "_").replace("/", "_")
        summary_path = os.path.join(output_dir, f"{safe_label}_analysis_summary.txt")
        generate_text_summary(analysis_results, summary_path)

    except ValueError as ve:
        logging.error(f"Error processing {label}: {ve}")
    except IOError as ioe:
        logging.error(f"Error during file operation for {label}: {ioe}")
    except ImportError as ie:
        logging.error(f"Import Error: Make sure all dependencies are installed ({ie})")
    except Exception as e:
        logging.error(f"An unexpected error occurred during analysis for {label}: {e}", exc_info=True)

    logging.info(f"--- Analysis complete for: {label} ---\n")


if __name__ == "__main__":
    logging.info("Starting SIGIL Benchmark Run...")
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
        entropy_data = get_entropy_data(source_label, DATA_SIZE_BYTES)

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
