import numpy as np
import math

def bytes_to_bit_array(data: bytes, width: int, height: int):
    """Converts bytes into a 2D numpy array of bits (0 or 1).

    Args:
        data: The raw bytes of entropy data.
        width: The desired width of the array.
        height: The desired height of the array.

    Returns:
        A 2D numpy array (height x width) of uint8 bits (0 or 1).

    Raises:
        ValueError: If insufficient data is provided for the specified dimensions.
    """
    num_pixels = width * height
    num_bits_needed = num_pixels
    num_bytes_needed = math.ceil(num_bits_needed / 8)

    if len(data) < num_bytes_needed:
        raise ValueError(f"Insufficient data: Need {num_bytes_needed} bytes for a {width}x{height} array, got {len(data)}.")

    # Extract needed bytes
    data_bytes = data[:num_bytes_needed]

    # Convert bytes to a flat list of bits
    bits = []
    for byte in data_bytes:
        bits.extend([(byte >> i) & 1 for i in range(7, -1, -1)]) # MSB first
    
    # Trim excess bits if num_bits_needed is not a multiple of 8
    bits = bits[:num_bits_needed]
    
    # Reshape into 2D array
    bit_array = np.array(bits, dtype=np.uint8).reshape((height, width))
    return bit_array

# TODO: Add other helper functions like entropy calculation, statistical tests etc.
