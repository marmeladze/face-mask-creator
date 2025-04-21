import logging
from .mask_creator import MaskCreator
from .utils import ensure_model_files

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure model files are downloaded
success, message = ensure_model_files()
if not success:
    logger.warning(f"Failed to download some model files: {message}")
else:
    logger.info(f"Model files status: {message}")

__version__ = '0.1.0'
__all__ = ['MaskCreator'] 