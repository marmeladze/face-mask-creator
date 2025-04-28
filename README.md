# Face Mask Creator

A simple library for creating face masks from images.

## Installation

### From PyPI (Recommended)

```bash
pip install face-mask-creator
```

After installation, you'll need to download the required model files. Run:
```bash
python -m face_mask_creator.setup
```

### From Source

1. Clone the repository:
```bash
git clone https://github.com/marmeladze/face-mask-creator.git
cd face-mask-creator
```

2. Install the package:
```bash
# Basic installation
pip install -e .

# With web interface
pip install -e ".[web]"
```

3. Download the required model files:
```bash
python setup.py
```

### From GitHub Release

```bash
pip install git+https://github.com/marmeladze/face-mask-creator.git@v0.0.3
```

After installation, download the required model files:
```bash
python -m face_mask_creator.setup
```

### Model Files

The library requires two model files:

1. **Shape Predictor Model**: `shape_predictor_68_face_landmarks.dat`
   - Source: [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
   - Used for: Detecting facial landmarks

2. **BiSeNet Face Parsing Model**: `bisenet_face_parsing.pth`
   - Source: [Google Drive](https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view)
   - Used for: Face parsing and segmentation

During the model setup, you'll be prompted to:
1. Choose whether to download the models or provide your own paths
2. If downloading, the models will be automatically downloaded and placed in the correct location
3. If using custom paths, you can specify them during setup

You can also use command-line options during setup:

```bash
# Use custom model files
python setup.py --shape-predictor /path/to/your/shape_predictor.dat --bisenet-model /path/to/your/bisenet.pth

# Skip downloading default models
python setup.py --skip-download
```

## Usage

### Python API

```python
from face_mask_creator import MaskCreator

# Initialize the creator
creator = MaskCreator()

# Create a mask from an image
mask = creator.create("path/to/image.jpg", output_type='binary')

# Save the mask
cv2.imwrite("output_mask.png", mask * 255)  # Scale to 0-255 for better visibility
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
- numpy>=1.19.0
- opencv-python>=4.5.0
- dlib>=19.24.6
- torch>=1.7.0
- torchvision>=0.8.0
- pillow>=8.0.0
- gdown>=4.7.1

### Web Interface Requirements (optional)
- flask>=2.0.0
- werkzeug>=2.0.0

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for a list of changes between versions.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 