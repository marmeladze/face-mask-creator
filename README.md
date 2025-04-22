# Face Mask Creator

A simple Python library for creating face masks from images using facial landmarks detection and BiSeNet face parsing.

## Installation

You can install the package directly from GitHub:

```bash
pip install git+https://github.com/marmeladze/face-mask-creator.git
```

The package will automatically download the required model files during installation.

## Usage

```python
from face_mask_creator import MaskCreator
import cv2

# Initialize the mask creator
mask_creator = MaskCreator()

# Read an image
image = cv2.imread('face.jpg')

# Create a binary mask
binary_mask = mask_creator.create(image, output_type='binary')

# Create a mask with original image
original_masked = mask_creator.create(image, output_type='original')

# Create a mask with skin-colored background
skin_masked = mask_creator.create(
    image,
    output_type='skin',
    skin_color=(255, 224, 189)  # Optional: specify skin color in RGB
)

# Get facial regions
regions = mask_creator.get_facial_regions(image)
if regions:
    for region_name, region_data in regions.items():
        print(f"{region_name}: {region_data['box']}")

# Save results
cv2.imwrite('binary_mask.png', binary_mask)
cv2.imwrite('original_masked.png', original_masked)
cv2.imwrite('skin_masked.png', skin_masked)
```

## Output Types

The library supports three types of output:

1. `binary`: A binary mask where 1 represents the face and 0 represents the background
2. `original`: The original image with the face mask applied
3. `skin`: The original image with the face mask applied and a skin-colored background

## Requirements

- Python 3.7 or higher
- numpy
- opencv-python
- dlib
- torch
- torchvision
- pillow

## Model Files

The library uses two model files, which are automatically downloaded during package installation:

1. dlib's facial landmarks predictor model: `shape_predictor_68_face_landmarks.dat`
2. BiSeNet face parsing model: `bisenet_face_parsing.pth`

These model files are stored in the package's `models` directory.

If you need to manually download the model files, you can find them at:
- dlib model: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- BiSeNet model: https://drive.google.com/uc?export=download&id=1-mEYsFwW5YQ2F6wJEbxk0OrKA9kQ5Z-_

## License

This project is licensed under the MIT License - see the LICENSE file for details. 