from setuptools import setup, find_packages
import os
from pathlib import Path
import urllib.request
import hashlib
import bz2
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# URLs for model files
MODEL_URLS = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "md5": "73fde5e05226548677a050913eed4e04",
        "compressed": True
    }
}

def download_model_files():
    """Download required model files during setup."""
    package_dir = Path(__file__).parent / "face_mask_creator"
    models_dir = package_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    for filename, info in MODEL_URLS.items():
        file_path = models_dir / filename
        if not file_path.exists():
            logger.info(f"Downloading {filename}...")
            compressed_path = file_path.with_suffix('.bz2') if info.get("compressed", False) else file_path
            
            # Download the file
            urllib.request.urlretrieve(info["url"], compressed_path)
            
            # Verify MD5 hash
            with open(compressed_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != info.get("md5"):
                raise ValueError(f"MD5 hash mismatch for {filename}")
            
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
    ],
    package_data={
        "face_mask_creator": ["models/*"],
    },
    py_modules=["face_mask_creator.utils"],
) 