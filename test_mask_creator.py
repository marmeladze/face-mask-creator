#!/usr/bin/env python3
"""
Test script for the face-mask-creator package.
"""

import os
import cv2
import numpy as np
from face_mask_creator import MaskCreator

def main():
    """Run a simple test of the MaskCreator class."""
    print("Testing face-mask-creator package...")
    
    # Initialize the mask creator
    print("Initializing MaskCreator...")
    mask_creator = MaskCreator()
    
    # Create a test image (a simple colored rectangle)
    print("Creating test image...")
    test_image = np.zeros((300, 300, 3), dtype=np.uint8)
    test_image[50:250, 50:250] = [0, 255, 0]  # Green rectangle
    
    # Try to create masks
    print("Creating masks...")
    try:
        binary_mask = mask_creator.create(test_image, output_type='binary')
        print(f"Binary mask shape: {binary_mask.shape}")
        
        original_masked = mask_creator.create(test_image, output_type='original')
        print(f"Original masked shape: {original_masked.shape}")
        
        skin_masked = mask_creator.create(test_image, output_type='skin')
        print(f"Skin masked shape: {skin_masked.shape}")
        
        # Try to get facial regions
        regions = mask_creator.get_facial_regions(test_image)
        if regions:
            print("Facial regions found:")
            for region_name, region_data in regions.items():
                print(f"  {region_name}: {region_data['box']}")
        else:
            print("No facial regions found (expected for a simple test image)")
        
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 