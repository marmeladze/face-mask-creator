from setuptools import setup, find_packages
import os
from pathlib import Path
import subprocess
import bz2
import logging
import sys
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URLs for model files
MODEL_URLS = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "compressed": True
    },
    "bisenet_face_parsing.pth": {
        "url": "https://github.com/zllrunning/face-parsing.PyTorch/raw/master/res/cp/79999_iter.pth",
        "compressed": False
    }
}

def parse_args():
    """Parse command line arguments."""
    # Only parse arguments if this script is run directly
    if len(sys.argv) > 1 and sys.argv[1] not in ['egg_info', 'install', 'develop']:
        parser = argparse.ArgumentParser(description='Setup face-mask-creator with optional custom model paths')
        parser.add_argument('--shape-predictor', type=str, help='Path to custom shape predictor model')
        parser.add_argument('--bisenet-model', type=str, help='Path to custom BiSeNet model')
        parser.add_argument('--skip-download', action='store_true', help='Skip downloading default models')
        return parser.parse_args()
    return None

def download_model_files(custom_paths=None, skip_download=False):
    """Download required model files during setup using wget."""
    package_dir = Path(__file__).parent / "face_mask_creator"
    models_dir = package_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    # Create a config file to store model paths
    config_dir = package_dir / "config"
    config_dir.mkdir(exist_ok=True)
    config_file = config_dir / "model_paths.json"
    
    model_paths = {}
    
    for filename, info in MODEL_URLS.items():
        file_path = models_dir / filename
        
        # Check if custom path is provided
        if custom_paths and filename in custom_paths and custom_paths[filename]:
            custom_path = Path(custom_paths[filename])
            if not custom_path.exists():
                logger.error(f"Custom model file not found: {custom_path}")
                sys.exit(1)
            model_paths[filename] = str(custom_path)
            logger.info(f"Using custom model for {filename}: {custom_path}")
            continue
            
        if skip_download and not file_path.exists():
            logger.warning(f"Skipping download of {filename} as requested")
            continue
            
        if not file_path.exists():
            logger.info(f"Downloading {filename}...")
            compressed_path = file_path.with_suffix('.bz2') if info.get("compressed", False) else file_path
            
            # Download the file using wget
            try:
                subprocess.run(["wget", info["url"], "-O", str(compressed_path)], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Error downloading {filename}: {e}")
                continue
            except FileNotFoundError:
                logger.error("wget not found. Please install wget or use a different download method.")
                continue
            
            # Extract if compressed
            if info.get("compressed", False):
                with bz2.BZ2File(compressed_path, 'rb') as source, open(file_path, 'wb') as target:
                    target.write(source.read())
                compressed_path.unlink()
            
            logger.info(f"Successfully downloaded and extracted {filename}")
        else:
            logger.info(f"{filename} already exists")
            
        model_paths[filename] = str(file_path)
    
    # Save model paths to config file
    import json
    with open(config_file, 'w') as f:
        json.dump(model_paths, f, indent=4)
    logger.info(f"Model paths saved to {config_file}")

# Parse command line arguments only if running directly
args = parse_args()

# Prepare custom paths dictionary if args exist
custom_paths = None
skip_download = False
if args:
    custom_paths = {
        "shape_predictor_68_face_landmarks.dat": args.shape_predictor,
        "bisenet_face_parsing.pth": args.bisenet_model
    }
    skip_download = args.skip_download

# Download model files
download_model_files(custom_paths=custom_paths, skip_download=skip_download)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-mask-creator",
    version="0.0.1",
    author="marmeladze",
    author_email="wrested@hotmail.de",
    description="A simple library for creating face masks from images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/marmeladze/face-mask-creator",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "dlib>=19.24.6",
        "torch>=1.7.0",
        "torchvision>=0.8.0",
        "pillow>=8.0.0",
    ],
    package_data={
        "face_mask_creator": ["models/*", "config/*"],
    },
    py_modules=["face_mask_creator.utils"],
) 