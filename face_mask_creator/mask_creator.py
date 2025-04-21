import cv2
import numpy as np
from typing import Dict, Optional, Tuple, Union
from .face_regions_parser import UnifiedFaceRegionsParser

class MaskCreator:
    """
    A simple interface for creating face masks from images.
    
    Example:
        ```python
        from face_mask_creator import MaskCreator
        
        # Initialize the mask creator
        mask_creator = MaskCreator()
        
        # Create a mask from an image
        image = cv2.imread('face.jpg')
        mask = mask_creator.create(image, output_type='binary')
        
        # Save the result
        cv2.imwrite('mask.png', mask)
        ```
    """
    
    def __init__(self):
        """Initialize the mask creator with the face regions parser."""
        self.face_parser = UnifiedFaceRegionsParser()
    
    def create(
        self,
        image: np.ndarray,
        output_type: str = 'binary',
        skin_color: Optional[Tuple[int, int, int]] = None
    ) -> np.ndarray:
        """
        Create a face mask from the input image.
        
        Args:
            image: Input image in BGR format (as returned by cv2.imread)
            output_type: Type of output mask. One of:
                - 'binary': Binary mask where 1 represents face and 0 represents background
                - 'original': Original image with face mask applied
                - 'skin': Original image with face mask applied and skin-colored background
            skin_color: RGB tuple for skin color (only used when output_type is 'skin')
                Default is light beige (255, 224, 189)
        
        Returns:
            numpy.ndarray: The created mask or masked image
        
        Raises:
            ValueError: If output_type is invalid
        """
        # Convert BGR to RGB for internal processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create base mask
        mask = self.face_parser.parse(image_rgb)
        
        if output_type == 'binary':
            return mask
            
        elif output_type == 'original':
            # Apply mask to original image
            masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            return cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            
        elif output_type == 'skin':
            if skin_color is None:
                skin_color = (255, 224, 189)  # Default light beige
                
            # Create colored background
            colored_bg = np.full_like(image_rgb, skin_color)
            
            # Apply mask
            masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
            inverse_mask = cv2.bitwise_not(mask)
            colored_bg = cv2.bitwise_and(colored_bg, colored_bg, mask=inverse_mask)
            result = cv2.add(masked_image, colored_bg)
            
            return cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
            
        else:
            raise ValueError(f"Invalid output_type: {output_type}. Must be one of: binary, original, skin")
    
    def get_facial_regions(self, image: np.ndarray) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Get facial regions from the image.
        
        Args:
            image: Input image in BGR format (as returned by cv2.imread)
            
        Returns:
            Dictionary containing bounding boxes for different facial regions.
            Returns None if no face is detected.
        """
        # Convert BGR to RGB for internal processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return self.face_parser.get_facial_regions(image_rgb) 