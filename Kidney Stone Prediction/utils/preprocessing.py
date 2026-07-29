import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

def apply_clahe(img_np, clip_limit=2.0, tile_grid_size=(8, 8)):
    """Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to RGB or grayscale numpy array."""
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        # Convert RGB to LAB space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(img_np)

def apply_denoise(img_np, filter_type='gaussian', kernel_size=3):
    """Applies Gaussian or Median Denoising Filter to reduce speckle noise."""
    if filter_type == 'gaussian':
        return cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
    elif filter_type == 'median':
        return cv2.medianBlur(img_np, kernel_size)
    return img_np

def get_transforms(target_size=(224, 224), augment=False):
    """Returns PyTorch Transform composition for CT scan images."""
    if augment:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def preprocess_input_image(pil_img, apply_clahe_flag=True, denoise_type='gaussian'):
    """Preprocesses a raw PIL image for inference."""
    img_np = np.array(pil_img.convert('RGB'))
    
    if denoise_type and denoise_type != 'none':
        img_np = apply_denoise(img_np, filter_type=denoise_type)
        
    if apply_clahe_flag:
        img_np = apply_clahe(img_np)
        
    processed_pil = Image.fromarray(img_np)
    transform = get_transforms(augment=False)
    tensor_img = transform(processed_pil).unsqueeze(0) # Batch dimension
    return tensor_img, img_np
