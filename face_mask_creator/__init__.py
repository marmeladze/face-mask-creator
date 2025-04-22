import logging
from .mask_creator import MaskCreator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__version__ = '0.1.0'
__all__ = ['MaskCreator'] 