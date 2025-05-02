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
    num_pixels_target = width * height

    # Check if we have at least enough bytes for the target pixels
    min_bytes_needed = math.ceil(num_pixels_target / 8)
    if len(data) < min_bytes_needed:
         raise ValueError(f"Insufficient data: Need at least {min_bytes_needed} bytes for {width}x{height} image, got {len(data)}.")

    # Unpack ALL provided bytes using numpy (LSB-first by default)
    bits_array_flat = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
    total_bits_available = len(bits_array_flat)

    if total_bits_available < num_pixels_target:
        # This shouldn't happen if the initial byte check passed, but safety check
        raise ValueError(f"Unpacking resulted in fewer bits ({total_bits_available}) than needed ({num_pixels_target}).")

    # Trim the flat BIT array to the exact number of pixels needed
    bits_to_plot = bits_array_flat[:num_pixels_target]
    
    # Reshape into 2D array
    bit_array = bits_to_plot.reshape((height, width))
    return bit_array.astype(np.uint8) # Ensure correct dtype

# TODO: Add other helper functions like entropy calculation, statistical tests etc.
