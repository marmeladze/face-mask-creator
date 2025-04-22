import cv2
import numpy as np
import dlib
import logging
import torch
import torchvision.transforms as transforms
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from PIL import Image
from .bisenet import BiSeNet

logger = logging.getLogger(__name__)

class FaceRegionsParser:
    """
    A class for parsing facial regions using dlib's facial landmarks.
    """
    def __init__(self, predictor_path: Optional[str] = None):
        """
        Initialize the face regions parser.

        Args:
            predictor_path (str, optional): Path to dlib's shape predictor file.
                If None, uses the default path from the package's models directory.
        """
        if predictor_path is None:
            # Use the predictor file from the package's models directory
            package_dir = Path(__file__).parent
            predictor_path = str(package_dir / "models" / "shape_predictor_68_face_landmarks.dat")
            
        logger.info(f"Initializing FaceRegionsParser with predictor: {predictor_path}")
        
        # Initialize dlib's face detector and facial landmarks predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        
        logger.info("FaceRegionsParser initialized successfully")

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract facial landmarks from the image.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Array of (x, y) coordinates for 68 facial landmarks.
                       Returns None if no face is detected.
        """
        # Convert to grayscale for dlib
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Detect faces
        faces = self.detector(gray)
        
        if len(faces) == 0:
            logger.warning("No face detected in the image")
            return None
            
        # Get landmarks for the first face
        face = faces[0]
        landmarks = self.predictor(gray, face)
        
        # Convert to numpy array
        landmarks_array = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(68)])
        
        return landmarks_array

    def get_forehead_region(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract forehead region.

        Args:
            landmarks (np.ndarray): Array of (x, y) coordinates for 68 facial landmarks.

        Returns:
            dict: Bounding box for the forehead region.
        """
        # Use landmarks 17-26 for forehead (eyebrows)
        left_point = landmarks[17]
        right_point = landmarks[26]
        
        # Estimate top of forehead (above eyebrows)
        vertical_offset = landmarks[27][1] - landmarks[51][1]  # Distance from nose to chin
        top_y = min(left_point[1], right_point[1]) - vertical_offset
        
        box = np.array([
            [left_point[0], top_y],           # top-left
            [right_point[0], top_y],          # top-right
            [right_point[0], landmarks[27][1]], # bottom-right
            [left_point[0], landmarks[27][1]]   # bottom-left
        ])
        
        return {'box': box}

    def get_eyes_region(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract eyes region (including crow's feet).

        Args:
            landmarks (np.ndarray): Array of (x, y) coordinates for 68 facial landmarks.

        Returns:
            dict: Bounding box for the eyes region.
        """
        # Left eye landmarks: 36-41
        # Right eye landmarks: 42-47
        left_eye_left = landmarks[36]
        left_eye_right = landmarks[39]
        right_eye_left = landmarks[42]
        right_eye_right = landmarks[45]
        
        # Add some padding for crow's feet
        padding = 20
        
        box = np.array([
            [left_eye_left[0] - padding, min(left_eye_left[1], right_eye_left[1]) - padding],  # top-left
            [right_eye_right[0] + padding, min(left_eye_left[1], right_eye_left[1]) - padding],  # top-right
            [right_eye_right[0] + padding, max(left_eye_right[1], right_eye_right[1]) + padding],  # bottom-right
            [left_eye_left[0] - padding, max(left_eye_right[1], right_eye_right[1]) + padding]  # bottom-left
        ])
        
        return {'box': box}

    def get_lips_region(self, landmarks: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract lips region.

        Args:
            landmarks (np.ndarray): Array of (x, y) coordinates for 68 facial landmarks.

        Returns:
            dict: Bounding box for the lips region.
        """
        # Lips landmarks: 48-67
        top_point = landmarks[51]  # Top of upper lip
        left_point = landmarks[48]  # Left corner of mouth
        right_point = landmarks[54]  # Right corner of mouth
        bottom_point = landmarks[57]  # Bottom of lower lip
        
        # Add some padding
        padding = 10
        
        box = np.array([
            [left_point[0] - padding, top_point[1] - padding],    # top-left
            [right_point[0] + padding, top_point[1] - padding],   # top-right
            [right_point[0] + padding, bottom_point[1] + padding], # bottom-right
            [left_point[0] - padding, bottom_point[1] + padding]   # bottom-left
        ])
        
        return {'box': box}

    def get_facial_regions(self, image: np.ndarray) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Extract all facial regions from the image.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            dict: Dictionary containing bounding boxes for different facial regions.
                 Returns None if no face is detected.
        """
        landmarks = self.get_landmarks(image)
        
        if landmarks is None:
            return None
            
        return {
            'forehead': self.get_forehead_region(landmarks),
            'eyes': self.get_eyes_region(landmarks),
            'lips': self.get_lips_region(landmarks)
        }

    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for the face.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Binary mask where 1 represents the face and 0 represents the background.
        """
        landmarks = self.get_landmarks(image)
        
        if landmarks is None:
            # Return empty mask if no face detected
            return np.zeros(image.shape[:2], dtype=np.uint8)
            
        # Create a mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Get facial regions
        regions = self.get_facial_regions(image)
        
        if regions is None:
            return mask
            
        # Draw regions on mask
        for region_name, region_data in regions.items():
            box = region_data['box'].astype(np.int32)
            cv2.fillPoly(mask, [box], 1)
            
        return mask


class BiSeNetFaceParser:
    """
    A class for parsing facial regions using BiSeNet model.
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the BiSeNet face parser.

        Args:
            model_path (str, optional): Path to pretrained BiSeNet model weights.
                If None, uses the default path from the package's models directory.
        """
        if model_path is None:
            # Use the model file from the package's models directory
            package_dir = Path(__file__).parent
            model_path = str(package_dir / "models" / "bisenet_face_parsing.pth")
            
        logger.info(f"Initializing BiSeNetFaceParser with model: {model_path}")
        
        # Initialize BiSeNet model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
        self.model = BiSeNet(n_classes=19)
        
        if Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("BiSeNet model loaded successfully")
        else:
            logger.warning(f"BiSeNet model file not found at {model_path}. Using uninitialized model.")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        logger.info("BiSeNetFaceParser initialized successfully")

    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Create a binary mask for the face using BiSeNet.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Binary mask where 1 represents the face and 0 represents the background.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        # Preprocess image
        input_tensor = self.transform(image).unsqueeze(0)
        input_tensor = input_tensor.to(self.device)
        
        with torch.no_grad():
            feat_out, _, _ = self.model(input_tensor)
            parsing = feat_out.squeeze(0).cpu().numpy().argmax(0)
            
        # Create binary mask for face and nose regions
        mask = np.zeros_like(parsing)
        mask[parsing == 1] = 1  # face
        mask[parsing == 10] = 1  # nose
        
        # Resize mask back to original image size
        mask = Image.fromarray(mask.astype(np.uint8))
        mask = mask.resize(image.size, Image.NEAREST)
        
        return np.array(mask)

    def get_facial_regions(self, image: np.ndarray) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Extract facial regions from the parsed face mask.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            dict: Dictionary containing bounding boxes for different facial regions.
                 Returns None if no face is detected.
        """
        face_mask = self.parse(image)
        height, width = face_mask.shape

        # Define regions based on the face mask
        regions = {
            'forehead': {
                'box': np.array([
                    [0, 0],                # top-left
                    [width, 0],            # top-right
                    [width, height // 3],  # bottom-right
                    [0, height // 3]       # bottom-left
                ])
            },
            'lips': {
                'box': np.array([
                    [width // 4, height // 2],      # top-left
                    [3 * width // 4, height // 2],  # top-right
                    [3 * width // 4, 3 * height // 4],  # bottom-right
                    [width // 4, 3 * height // 4]       # bottom-left
                ])
            },
            'crows_feet': {
                'box': np.array([
                    [0, height // 3],      # top-left
                    [width, height // 3],  # top-right
                    [width, height // 2],  # bottom-right
                    [0, height // 2]       # bottom-left
                ])
            }
        }
        return regions


class UnifiedFaceRegionsParser:
    """
    A unified face regions parser that can use different backends.
    """
    def __init__(self, use_bisenet: bool = True):
        """
        Initialize the unified face regions parser.
        
        Args:
            use_bisenet (bool): Whether to use BiSeNet model for face parsing.
                If False, uses dlib's facial landmarks.
        """
        if use_bisenet:
            self.parser = BiSeNetFaceParser()
        else:
            self.parser = FaceRegionsParser()
        
    def parse(self, image: np.ndarray) -> np.ndarray:
        """
        Parse the face mask.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            np.ndarray: Binary face mask.
        """
        return self.parser.parse(image)
        
    def get_facial_regions(self, image: np.ndarray) -> Optional[Dict[str, Dict[str, np.ndarray]]]:
        """
        Extract facial regions from the image.

        Args:
            image (np.ndarray): Input image in RGB format.

        Returns:
            dict: Dictionary containing bounding boxes for different facial regions.
                 Returns None if no face is detected.
        """
        return self.parser.get_facial_regions(image) 