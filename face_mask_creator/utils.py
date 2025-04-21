import os
import logging
import urllib.request
import hashlib
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# URLs for model files
MODEL_URLS = {
    "shape_predictor_68_face_landmarks.dat": {
        "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
        "md5": "73fde5e05226548677a050913eed4e04",
        "compressed": True
    }
}

def download_file(url: str, destination: Path, md5_hash: Optional[str] = None, compressed: bool = False) -> bool:
    """
    Download a file from a URL to a destination path.
    
    Args:
        url: URL to download from
        destination: Path to save the file to
        md5_hash: Expected MD5 hash of the file (optional)
        compressed: Whether the file is compressed (bz2)
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        # Create parent directories if they don't exist
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the file
        logger.info(f"Downloading {url} to {destination}")
        urllib.request.urlretrieve(url, destination)
        
        # Verify MD5 hash if provided
        if md5_hash:
            with open(destination, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash != md5_hash:
                logger.error(f"MD5 hash mismatch for {destination}")
                return False
            logger.info(f"MD5 hash verified for {destination}")
        
        # Extract if compressed
        if compressed and str(destination).endswith('.bz2'):
            import bz2
            with bz2.BZ2File(destination, 'rb') as source, open(destination.with_suffix(''), 'wb') as target:
                target.write(source.read())
            # Remove the compressed file
            destination.unlink()
            logger.info(f"Extracted {destination.with_suffix('')}")
        
        return True
    except Exception as e:
        logger.error(f"Error downloading {url}: {e}")
        return False

def ensure_model_files() -> Tuple[bool, str]:
    """
    Ensure all required model files are downloaded.
    
    Returns:
        Tuple[bool, str]: (success, message)
    """
    package_dir = Path(__file__).parent
    models_dir = package_dir / "models"
    
    # Create models directory if it doesn't exist
    models_dir.mkdir(exist_ok=True)
    
    all_success = True
    messages = []
    
    for filename, info in MODEL_URLS.items():
        file_path = models_dir / filename
        if not file_path.exists():
            success = download_file(
                info["url"], 
                file_path.with_suffix('.bz2') if info.get("compressed", False) else file_path,
                info.get("md5"),
                info.get("compressed", False)
            )
            if not success:
                all_success = False
                messages.append(f"Failed to download {filename}")
            else:
                messages.append(f"Successfully downloaded {filename}")
        else:
            messages.append(f"{filename} already exists")
    
    return all_success, "\n".join(messages) 