# Face Mask Creator

A simple library for creating face masks from images.

## Installation

### Basic Installation

```bash
pip install -e .
```

### With Web Interface

To install with the web interface, use:

```bash
pip install -e ".[web]"
```

This will install additional dependencies required for the web server.

### Model Files

The library requires two model files:

1. **Shape Predictor Model**: `shape_predictor_68_face_landmarks.dat`
   - Source: [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Used for: Detecting facial landmarks

2. **BiSeNet Face Parsing Model**: `bisenet_face_parsing.pth`
   - Source: [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth)
   - Used for: Face parsing and segmentation

These models will be automatically downloaded during installation. If you want to use your own model files or skip the download, you can use the following options:

```bash
# Use custom model files
python setup.py --shape-predictor /path/to/your/shape_predictor.dat --bisenet-model /path/to/your/bisenet.pth

# Skip downloading default models
python setup.py --skip-download
```

## Usage

### Python API

```python
from face_mask_creator import FaceMaskCreator

# Initialize the creator
creator = FaceMaskCreator()

# Create a mask from an image
mask = creator.create_mask("path/to/image.jpg")

# Save the mask
mask.save("output_mask.png")
```

### Web Interface

If you installed with the web extras, you can start the web server:

```python
from face_mask_creator.extras.web import start_server

# Start the server (default: host='0.0.0.0', port=5000)
start_server()

# Or with custom host and port
start_server(host='localhost', port=8080)
```

Then open your browser and navigate to `http://localhost:5000` (or your custom port) to use the web interface.

## Requirements

### Core Requirements
- Python 3.7+
- numpy
- opencv-python
- dlib
- torch
- torchvision
- pillow

### Web Interface Requirements (optional)
- flask>=2.0.0
- werkzeug>=2.0.0

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes between versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 