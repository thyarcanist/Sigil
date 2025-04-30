import os
import requests
import logging
from typing import Optional
from dotenv import load_dotenv
import time # Added for delays
import base64 # Added for decoding
import math # Added for ceil

# Load environment variables from .env file
load_dotenv(override=True) # Force override of existing env vars

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Robust Fetching ---
# MAX_BYTES_PER_REQUEST = 512 * 1024  # 512 KiB chunk size (adjust if needed) - Still too large
# MAX_BYTES_PER_REQUEST = 256 * 1024  # Try 256 KiB chunk size - Still too large
MAX_BYTES_PER_REQUEST = 120 * 1024  # Set below documented ~124KB limit
MAX_RETRIES = 3
REQUEST_TIMEOUT = 180 # seconds (Increased timeout)
RETRY_DELAY_SECONDS = 2
CHUNK_DELAY_SECONDS = 1
MIN_REQUEST_SIZE_WORKAROUND = 24 # From user code example
# ------------------------------------

def _get_api_config() -> tuple[Optional[str], Optional[str]]:
    """Retrieves API URL and Key from environment variables."""
    # Use OCCYBYTE_ prefixed variables
    api_url = os.environ.get("OCCYBYTE_API_URL")
    api_key = os.environ.get("OCCYBYTE_API_KEY")

    if not api_url:
        logging.warning("OCCYBYTE_API_URL environment variable not set (checked .env and environment).")
    if not api_key:
        logging.warning("OCCYBYTE_API_KEY environment variable not set (checked .env and environment). API calls requiring auth will likely fail.")
        api_key = "YOUR_API_KEY_HERE" # Prevent None type error later, but log warning

    return api_url, api_key

# --- Robust Quantum Byte Fetching (Adapted from User Code) ---
def fetch_entropy_from_api(endpoint: str, total_bytes_needed: int) -> Optional[bytes]:
    """
    Fetches entropy data from a specified QuantumDataService API endpoint,
    handling chunking, retries, and potential base64 padding errors.

    Args:
        endpoint: The specific API endpoint path (e.g., '/api/eris/invoke' or '/api/eris/raw').
        total_bytes_needed: The total number of bytes of entropy to request.

    Returns:
        The fetched entropy data as bytes, or None if an error occurs.
    """
    api_url_base, api_key = _get_api_config()

    if not api_url_base:
        logging.error("API URL is not configured. Cannot fetch entropy.")
        return None
    if not api_key or api_key == "YOUR_API_KEY_HERE":
        logging.error("API Key not configured or is placeholder. Cannot fetch entropy.")
        return None
    if total_bytes_needed <= 0:
        logging.info("Requested 0 or fewer bytes, returning empty bytes.")
        return b''

    # --- Determine Chunk Size Based on Endpoint ---
    # /invoke (whitened) is more intensive, use smaller chunks
    if endpoint == '/api/eris/invoke':
        current_max_bytes_per_request = 60 * 1024 # 60 KiB for invoke
        logging.info(f"Using smaller chunk size ({current_max_bytes_per_request} bytes) for {endpoint}")
    else:
        current_max_bytes_per_request = MAX_BYTES_PER_REQUEST # Use default (120 KiB) for others
    # --------------------------------------------

    # --- Workaround for potential API base64 padding errors on small requests ---
    # Only apply if the total needed is small AND fits within one chunk.
    bytes_to_actually_request = total_bytes_needed
    # Use the endpoint-specific chunk size for this check
    if total_bytes_needed < MIN_REQUEST_SIZE_WORKAROUND and total_bytes_needed < current_max_bytes_per_request:
        bytes_to_actually_request = MIN_REQUEST_SIZE_WORKAROUND
        logging.info(f"Workaround: Adjusted request size from {total_bytes_needed} to {bytes_to_actually_request} to potentially avoid API padding errors.")

    all_fetched_bytes = bytearray()
    # Calculate num_chunks based on the potentially adjusted request size and endpoint-specific chunk size
    num_chunks = math.ceil(bytes_to_actually_request / current_max_bytes_per_request)
    base_url = api_url_base.rstrip('/')
    full_endpoint_base = f"{base_url}{endpoint}"

    logging.info(f"Starting fetch for {total_bytes_needed} bytes (requesting {bytes_to_actually_request}) from {endpoint} in {num_chunks} chunk(s) of max size {current_max_bytes_per_request}...") # Log max size

    for i in range(num_chunks):
        # Calculate size for this chunk based on the *potentially adjusted* total request size
        bytes_remaining_to_request = bytes_to_actually_request - len(all_fetched_bytes)
        # Use endpoint-specific chunk size here
        bytes_to_request_this_chunk = min(bytes_remaining_to_request, current_max_bytes_per_request)

        if bytes_to_request_this_chunk <= 0:
            logging.info("All requested bytes seem to have been fetched in previous chunks.")
            break # Should not happen if num_chunks is calculated correctly, but safety first

        request_url = f"{full_endpoint_base}?size={bytes_to_request_this_chunk}"
        headers = {"X-API-Key": api_key}

        success = False
        for attempt in range(MAX_RETRIES + 1):
            try:
                logging.info(f"  Requesting chunk {i+1}/{num_chunks} ({bytes_to_request_this_chunk} bytes) from {request_url}, attempt {attempt+1}/{MAX_RETRIES+1}...")
                response = requests.get(request_url, headers=headers, timeout=REQUEST_TIMEOUT)
                response.raise_for_status() # Check for HTTP errors (4xx, 5xx)

                json_response = response.json()
                if "data" not in json_response:
                     raise ValueError("'data' field not found in API response")

                base64_data = json_response["data"]
                # Try decoding early to catch padding errors
                try:
                    chunk_bytes = base64.b64decode(base64_data)
                    # API might return slightly more than requested due to base64 alignment, that's okay.
                except base64.binascii.Error as decode_error:
                    # Re-raise as ValueError for retry logic
                    raise ValueError(f"Base64 decode failed: {decode_error}. API likely returned malformed data for size {bytes_to_request_this_chunk}.")

                # Store fetched bytes
                all_fetched_bytes.extend(chunk_bytes)
                logging.info(f"  Received {len(chunk_bytes)} bytes. Total fetched so far: {len(all_fetched_bytes)}")
                success = True
                break # Exit retry loop on success

            except requests.exceptions.RequestException as e: # Network errors, timeout, etc.
                logging.warning(f"  Attempt {attempt+1} failed (RequestException): {e}")
            except ValueError as e: # JSON errors, missing 'data', decode errors, insufficient bytes returned
                 logging.warning(f"  Attempt {attempt+1} failed (ValueError): {e}")
            except Exception as e: # Catch-all for unexpected errors during request/processing
                logging.warning(f"  Attempt {attempt+1} failed (Unexpected Error): {e}", exc_info=False) # Show short error in log

            # If not successful after this attempt
            if not success:
                if attempt < MAX_RETRIES:
                    logging.info(f"  Waiting {RETRY_DELAY_SECONDS}s before retrying...")
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"Chunk {i+1} failed after {MAX_RETRIES+1} attempts. Aborting fetch for {endpoint}.")
                    return None # Failed to get this chunk after all retries

        if not success:
             return None # Exit outer loop if a chunk failed permanently

        # Optional delay between chunks if needed
        if success and i < num_chunks - 1:
            logging.info(f"Waiting {CHUNK_DELAY_SECONDS}s before next chunk...")
            time.sleep(CHUNK_DELAY_SECONDS)

    # --- Final Check and Slice ---
    # Check if we got *at least* the amount *originally* needed
    if len(all_fetched_bytes) < total_bytes_needed:
        logging.error(f"Final fetched bytes ({len(all_fetched_bytes)}) less than originally required ({total_bytes_needed}) for {endpoint}.")
        return None

    logging.info(f"Successfully fetched {len(all_fetched_bytes)} total bytes for {endpoint} (originally needed {total_bytes_needed}).")
    # Slice the result to return *exactly* the originally requested number of bytes
    return bytes(all_fetched_bytes[:total_bytes_needed])
# -------------------------------------------------------------

def fetch_eris_full(size_bytes: int) -> Optional[bytes]:
    """Fetches whitened quantum random data (eris:full) from the API."""
    return fetch_entropy_from_api('/api/eris/invoke', size_bytes)

def fetch_eris_raw(size_bytes: int) -> Optional[bytes]:
    """Fetches raw, unwhitened quantum random data (eris:raw) from the API."""
    return fetch_entropy_from_api('/api/eris/raw', size_bytes)

# Example Usage (if run directly)
if __name__ == "__main__":
    print("Testing API client utility...")
    # Note: This now requires OCCYBYTE_API_URL and OCCYBYTE_API_KEY to be set
    # either in the environment or in a .env file in the project root.
    test_size = 32 # Small test
    large_test_size = 600 * 1024 # Test chunking

    print(f"\nAttempting to fetch {test_size} bytes of eris:full...")
    full_data = fetch_eris_full(test_size)
    if full_data:
        print(f"  Success! Received {len(full_data)} bytes. Hex: {full_data.hex()}")
    else:
        print("  Failed to fetch eris:full data. Check logs and environment variables (OCCYBYTE_API_URL, OCCYBYTE_API_KEY in .env or env).")

    print(f"\nAttempting to fetch {test_size} bytes of eris:raw...")
    raw_data = fetch_eris_raw(test_size)
    if raw_data:
        print(f"  Success! Received {len(raw_data)} bytes. Hex: {raw_data.hex()}")
    else:
        print("  Failed to fetch eris:raw data. Check logs and environment variables.") 

    # Test Chunking
    print(f"\nAttempting to fetch {large_test_size} bytes of eris:raw (testing chunking)...")
    large_raw_data = fetch_eris_raw(large_test_size)
    if large_raw_data:
         print(f"  Chunking Success! Received {len(large_raw_data)} bytes.")
    else:
         print("  Failed to fetch large eris:raw data using chunking.") 