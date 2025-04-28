from setuptools import setup, find_packages
import os
from pathlib import Path
import subprocess
import bz2
import logging
import sys
import argparse
import json

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
        "url": "https://drive.google.com/file/d/154JgKpzCPW82qINcVieuPH3fZ2e0P812/view",
        "compressed": False
    }
}

def get_user_input(prompt, default=None):
    """Get user input with optional default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def download_from_gdrive(file_id, output_path):
    """Download a file from Google Drive using gdown."""
    try:
        import gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)
        return True
    except Exception as e:
        logger.error(f"Error downloading from Google Drive: {e}")
        return False

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
    
    # Check if we're running in a pip install context
    is_pip_install = any(arg.startswith(('install', 'bdist', 'build')) for arg in sys.argv[1:])
    
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
            # During pip install, skip interactive prompts
            if is_pip_install:
                logger.info(f"Skipping interactive model download during pip install for {filename}")
                logger.info("Please run 'python setup.py' after installation to download models")
                continue
                
            # Ask user for input
            print(f"\nModel file '{filename}' not found.")
            choice = get_user_input(
                "Do you want to:\n"
                "1. Download from internet\n"
                "2. Provide path to existing file\n"
                "3. Skip this model\n"
                "Enter choice (1-3)",
                "1"
            )
            
            if choice == "1":
                logger.info(f"Downloading {filename}...")
                if filename == "bisenet_face_parsing.pth":
                    # Extract file ID from Google Drive URL
                    file_id = info["url"].split("/")[-2]
                    if not download_from_gdrive(file_id, str(file_path)):
                        logger.error(f"Failed to download {filename}")
                        continue
                else:
                    compressed_path = file_path.with_suffix('.bz2') if info.get("compressed", False) else file_path
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
            
            elif choice == "2":
                custom_path = get_user_input("Enter path to existing model file")
                if not os.path.exists(custom_path):
                    logger.error(f"File not found: {custom_path}")
                    continue
                model_paths[filename] = custom_path
                logger.info(f"Using custom model file: {custom_path}")
            
            else:
                logger.warning(f"Skipping {filename}")
                continue
        else:
            logger.info(f"{filename} already exists")
            
        if file_path.exists():
            model_paths[filename] = str(file_path)
    
    # Save model paths to config file
    with open(config_file, 'w') as f:
        json.dump(model_paths, f, indent=4)
    logger.info(f"Model paths saved to {config_file}")

def run_smoke_tests():
    """Run smoke tests to verify installation."""
    logger.info("Running smoke tests...")
    
    try:
        # Test importing required packages
        import cv2
        import numpy as np
        import dlib
        import torch
        import torchvision
        from PIL import Image
        
        # Test importing the package
        from face_mask_creator import MaskCreator
        
        # Test model loading
        creator = MaskCreator()
        
        # Create a test image
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        test_image[25:75, 25:75] = [255, 255, 255]
        
        # Test mask creation
        mask = creator.create(test_image, output_type='binary')
        
        logger.info("Smoke tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Smoke tests failed: {e}")
        return False

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

# Run smoke tests
if not run_smoke_tests():
    logger.warning("Smoke tests failed. The installation may not be complete.")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="face-mask-creator",
    version="0.0.4",
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
        "gdown>=4.7.1",  # Added for Google Drive downloads
    ],
    extras_require={
        'web': [
            'flask>=2.0.0',
            'werkzeug>=2.0.0',
        ],
    },
    package_data={
        "face_mask_creator": ["models/*", "config/*"],
        "face_mask_creator.extras.web": ["templates/*"],
    },
    py_modules=["face_mask_creator.utils"],
) 