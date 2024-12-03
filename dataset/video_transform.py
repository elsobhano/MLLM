import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from typing import List
import random
from PIL import Image

class ConsistentVideoTransforms:
    def __init__(self, mode: str = 'train'):
        self.mode = mode
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        self.s = 1  # scale factor for color jitter
        
    def get_random_params(self):
        """Generate random parameters for transforms once per video"""
        params = {}
        
        if self.mode == 'train':
            # Random rotation params (p=0.3)
            params['should_rotate'] = random.random() < 0.3
            params['rotation_angle'] = random.uniform(-5, 5) if params['should_rotate'] else 0
            
            # Random resized crop params
            params['crop_params'] = T.RandomResizedCrop.get_params(
                img=Image.new('RGB', (256, 256)),  # dummy image for getting params
                scale=(0.875, 1.0),
                ratio=(0.9, 1.1)
            )
            
            # Horizontal flip params (p=0.5)
            params['should_flip'] = random.random() < 0.5
            
            # Color jitter params (p=0.3)
            params['should_color_jitter'] = random.random() < 0.3
            if params['should_color_jitter']:
                params['brightness'] = random.uniform(1-0.8*self.s, 1+0.8*self.s)
                params['contrast'] = random.uniform(1-0.8*self.s, 1+0.8*self.s)
                params['saturation'] = random.uniform(1-0.8*self.s, 1+0.8*self.s)
                params['hue'] = random.uniform(-0.2*self.s, 0.2*self.s)
            
            # Grayscale params (p=0.2)
            params['should_gray'] = random.random() < 0.2
            
        return params
    
    def transform_frame(self, frame: Image.Image, params: dict) -> torch.Tensor:
        """Apply consistent transforms to a single frame using saved parameters"""
        # Resize to 256x256
        frame = F.resize(frame, (256, 256))
        
        if self.mode == 'train':
            # Apply rotation
            # if params['should_rotate']:
            #     frame = F.rotate(frame, params['rotation_angle'])
            
            # Apply random resized crop
            i, j, h, w = params['crop_params']
            frame = F.resized_crop(frame, i, j, h, w, (224, 224))
            
            # Apply horizontal flip
            if params['should_flip']:
                frame = F.hflip(frame)
            
            # Apply color jitter
            if params['should_color_jitter']:
                frame = F.adjust_brightness(frame, params['brightness'])
                frame = F.adjust_contrast(frame, params['contrast'])
                frame = F.adjust_saturation(frame, params['saturation'])
                frame = F.adjust_hue(frame, params['hue'])
            
            # Apply grayscale
            if params['should_gray']:
                frame = F.rgb_to_grayscale(frame, num_output_channels=3)
        else:
            # Evaluation mode: simple center crop
            frame = F.center_crop(frame, 224)
        
        # Convert to tensor and normalize
        frame = F.to_tensor(frame)
        frame = F.normalize(frame, self.mean, self.std)
        
        return frame
    
    def __call__(self, frames: List[Image.Image]) -> torch.Tensor:
        """
        Apply consistent transformations across all frames in a video sequence.
        
        Args:
            frames: List of PIL Images
        Returns:
            Transformed frames tensor [T, C, H, W]
        """
        # Generate random parameters once for the entire video
        params = self.get_random_params()
        
        # Apply the same transforms to all frames
        transformed_frames = [
            self.transform_frame(frame, params) for frame in frames
        ]
        
        return torch.stack(transformed_frames)