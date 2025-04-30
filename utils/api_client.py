import os
import requests
import logging
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _get_api_config() -> tuple[Optional[str], Optional[str]]:
    """Retrieves API URL and Key from environment variables."""
    # Use OCCYBYTE_ prefixed variables
    api_url = os.environ.get("OCCYBYTE_API_URL")
    api_key = os.environ.get("OCCYBYTE_API_KEY")

    if not api_url:
        logging.warning("OCCYBYTE_API_URL environment variable not set (checked .env and environment).")
    if not api_key:
        logging.warning("OCCYBYTE_API_KEY environment variable not set (checked .env and environment). API calls requiring auth will likely fail.")

    return api_url, api_key

def fetch_entropy_from_api(endpoint: str, size_bytes: int) -> Optional[bytes]:
    """
    Fetches entropy data from a specified QuantumDataService API endpoint.

    Args:
        endpoint: The specific API endpoint path (e.g., '/api/eris/invoke' or '/api/eris/raw').
        size_bytes: The number of bytes of entropy to request.

    Returns:
        The fetched entropy data as bytes, or None if an error occurs.
    """
    api_url_base, api_key = _get_api_config()

    if not api_url_base:
        logging.error("API URL is not configured. Cannot fetch entropy.")
        return None
    if not api_key:
        logging.error("API Key is not configured. Cannot fetch entropy.")
        return None
    if size_bytes <= 0:
        logging.error("Size must be a positive integer.")
        return None

    # Construct the full URL
    full_url = f"{api_url_base.rstrip('/')}{endpoint}"
    params = {'size': size_bytes}
    headers = {'Authorization': f'Bearer {api_key}'} # Assuming Bearer token auth

    logging.info(f"Fetching {size_bytes} bytes from {full_url}...")
    try:
        response = requests.get(full_url, params=params, headers=headers, timeout=120) # Increased timeout for potentially large requests
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        logging.info(f"Successfully fetched {len(response.content)} bytes.")
        return response.content

    except requests.exceptions.RequestException as e:
        logging.error(f"API request failed: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        return None

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
    test_size = 32

    print(f"Attempting to fetch {test_size} bytes of eris:full...")
    full_data = fetch_eris_full(test_size)
    if full_data:
        print(f"  Success! Received {len(full_data)} bytes. Hex: {full_data.hex()}")
    else:
        print("  Failed to fetch eris:full data. Check logs and environment variables (OCCYBYTE_API_URL, OCCYBYTE_API_KEY in .env or env).")

    print(f"Attempting to fetch {test_size} bytes of eris:raw...")
    raw_data = fetch_eris_raw(test_size)
    if raw_data:
        print(f"  Success! Received {len(raw_data)} bytes. Hex: {raw_data.hex()}")
    else:
        print("  Failed to fetch eris:raw data. Check logs and environment variables.") 