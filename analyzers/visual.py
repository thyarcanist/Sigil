import numpy as np
from PIL import Image
import math
import os
# Import the helper function
from ..utils.helpers import bytes_to_bit_array 

def generate_bit_visualization(data: bytes, width: int, height: int, output_path: str = None, scale: int = 1):
    """Generates a monochrome bitmap visualization of the entropy data.

    Args:
        data: The raw bytes of entropy data.
        width: The desired width of the visualization in pixels.
        height: The desired height of the visualization in pixels.
        output_path: Optional path to save the image file. If None, the PIL Image object is returned.
        scale: Integer factor to scale the image up by (e.g., 2 means each bit becomes 2x2 pixels).

    Returns:
        PIL.Image.Image object if output_path is None, otherwise None.
    """
    # Use the imported helper function
    bit_array = bytes_to_bit_array(data, width, height)

    # Create image (0=black, 255=white)
    # Map bit 0 -> black (0), bit 1 -> white (255)
    img_array = (bit_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array, mode='L')

    # Scale image if requested
    if scale > 1:
        img = img.resize((width * scale, height * scale), Image.NEAREST) # Use nearest neighbor for sharp pixels

    if output_path:
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            img.save(output_path)
            print(f"Bit visualization saved to: {output_path}")
            return None
        except Exception as e:
            raise IOError(f"Error saving image to {output_path}: {e}")
    else:
        return img

# Example usage (if run directly):
if __name__ == '__main__':
    # Generate some dummy random data
    dummy_data = os.urandom(1125000) # For a 3000x3000 image
    
    try:
        print("Generating 3000x3000 visualization...")
        generate_bit_visualization(dummy_data, 3000, 3000, './dummy_bit_viz_3000.png')
        
        print("Generating 100x100 visualization (scaled up x5)...")
        # Need less data for smaller image
        dummy_small_data = os.urandom(1250) # 100*100 / 8
        generate_bit_visualization(dummy_small_data, 100, 100, './dummy_bit_viz_100_scaled.png', scale=5)

        print("Generating 100x100 visualization (returned as object)...")
        img_obj = generate_bit_visualization(dummy_small_data, 100, 100)
        if img_obj:
            print(f"Returned Image object: {type(img_obj)}, Size: {img_obj.size}")
            # img_obj.show() # Uncomment to display the image if Pillow is configured

    except ValueError as ve:
        print(f"Error: {ve}")
    except IOError as ioe:
        print(f"Error: {ioe}") 