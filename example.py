#!/usr/bin/env python3
"""
Example script demonstrating the usage of the face-mask-creator package.
"""

import os
import cv2
import numpy as np
from face_mask_creator import MaskCreator

def main():
    """Run an example of the MaskCreator class."""
    print("Face Mask Creator Example")
    print("=========================")
    
    # Initialize the mask creator
    print("Initializing MaskCreator...")
    mask_creator = MaskCreator()
    
    # Check if an image file was provided as a command-line argument
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if not os.path.exists(image_path):
            print(f"Error: Image file '{image_path}' not found.")
            return False
    else:
        # Use a default image if available
        image_path = "example_face.jpg"
        if not os.path.exists(image_path):
            print(f"Error: Default image file '{image_path}' not found.")
            print("Please provide an image file as a command-line argument:")
            print(f"  python {sys.argv[0]} path/to/your/image.jpg")
            return False
    
    # Read the image
    print(f"Reading image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Failed to read image '{image_path}'.")
        return False
    
    # Create output directory
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create masks
    print("Creating masks...")
    
    # Binary mask
    binary_mask = mask_creator.create(image, output_type='binary')
    binary_path = os.path.join(output_dir, "binary_mask.png")
    cv2.imwrite(binary_path, binary_mask * 255)  # Scale to 0-255 for better visibility
    print(f"Binary mask saved to: {binary_path}")
    
    # Original masked
    original_masked = mask_creator.create(image, output_type='original')
    original_path = os.path.join(output_dir, "original_masked.png")
    cv2.imwrite(original_path, original_masked)
    print(f"Original masked image saved to: {original_path}")
    
    # Skin masked
    skin_masked = mask_creator.create(image, output_type='skin')
    skin_path = os.path.join(output_dir, "skin_masked.png")
    cv2.imwrite(skin_path, skin_masked)
    print(f"Skin masked image saved to: {skin_path}")
    
    # Get facial regions
    regions = mask_creator.get_facial_regions(image)
    if regions:
        print("\nFacial regions found:")
        for region_name, region_data in regions.items():
            print(f"  {region_name}: {region_data['box']}")
            
        # Draw regions on a copy of the original image
        regions_image = image.copy()
        for region_name, region_data in regions.items():
            box = region_data['box'].astype(np.int32)
            cv2.polylines(regions_image, [box], True, (0, 255, 0), 2)
            # Add region name
            cv2.putText(regions_image, region_name, 
                        (int(box[0][0]), int(box[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        regions_path = os.path.join(output_dir, "facial_regions.png")
        cv2.imwrite(regions_path, regions_image)
        print(f"Facial regions visualization saved to: {regions_path}")
    else:
        print("\nNo facial regions found in the image.")
    
    print("\nExample completed successfully!")
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 