import os

def load_from_binary_file(filepath: str) -> bytes:
    """Loads raw bytes from a specified binary file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Entropy file not found: {filepath}")
    
    try:
        with open(filepath, 'rb') as f:
            data = f.read()
        return data
    except Exception as e:
        raise IOError(f"Error reading entropy file {filepath}: {e}")

def load_from_hex_file(filepath: str) -> bytes:
    """Loads bytes from a file containing hexadecimal characters."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Hex entropy file not found: {filepath}")

    try:
        with open(filepath, 'r') as f:
            hex_string = f.read().replace("\n", "").replace(" ", "") # Remove whitespace/newlines
        # Validate hex string
        if not all(c in '0123456789abcdefABCDEF' for c in hex_string):
            raise ValueError("File contains non-hexadecimal characters.")
        # Ensure even length for correct byte conversion
        if len(hex_string) % 2 != 0:
            raise ValueError("Hex string must have an even number of characters.")
        
        return bytes.fromhex(hex_string)
    except ValueError as ve:
         raise ValueError(f"Invalid hex data in {filepath}: {ve}")
    except Exception as e:
        raise IOError(f"Error reading hex entropy file {filepath}: {e}")

# TODO: Add function to load from specific formats if needed (e.g., NIST format)
# TODO: Add function to load directly from an entropy source object/API call
