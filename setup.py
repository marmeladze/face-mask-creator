from setuptools import setup, find_packages
import os
from pathlib import Path
import subprocess
import bz2
import logging
import sys

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

def download_model_files():
    """Download required model files during setup using wget."""
    package_dir = Path(__file__).parent / "face_mask_creator"
    models_dir = package_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    for filename, info in MODEL_URLS.items():
        file_path = models_dir / filename
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

# Download model files
download_model_files()

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
        "face_mask_creator": ["models/*"],
    },
    py_modules=["face_mask_creator.utils"],
) 