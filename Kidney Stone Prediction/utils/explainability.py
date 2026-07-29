import cv2
import numpy as np
import torch
import torch.nn.functional as F

class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for PyTorch models."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._forward_hook)
        self.target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = torch.argmax(output, dim=1).item()
            
        self.model.zero_grad()
        score = output[0, target_class]
        score.backward(retain_graph=True)
        
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        # Global Average Pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted sum of activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam) # Normalize to [0, 1]
            
        cam = cv2.resize(cam, (input_tensor.shape[3], input_tensor.shape[2]))
        return cam, target_class, output

def overlay_heatmap_on_image(original_img_np, cam, alpha=0.5):
    """Overlays Grad-CAM heatmap on RGB numpy image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    if original_img_np.shape[:2] != heatmap.shape[:2]:
        heatmap = cv2.resize(heatmap, (original_img_np.shape[1], original_img_np.shape[0]))
        
    overlay = np.uint8(alpha * heatmap + (1 - alpha) * original_img_np)
    return overlay, heatmap

def detect_calcification_roi(img_np, intensity_threshold=210):
    """
    Detects high-density calcifications (kidney stones) using OpenCV morphology & contour detection.
    Returns segmented image with bounding box, area, and center coordinates.
    """
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if len(img_np.shape) == 3 else img_np
    
    # Threshold for high CT intensity (bright white calcification)
    _, thresh = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)
    
    # Morphological opening to clean noise
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    annotated_img = img_np.copy()
    detected_stones = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 8 <= area <= 500: # Filter out tiny noise and massive spine region
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Avoid central spine area (x between 95 and 125 near bottom)
            if 95 <= x <= 125 and y > 150:
                continue
                
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(annotated_img, f"Stone ({int(area)}px)", (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            detected_stones.append({
                'bbox': (x, y, w, h),
                'area': area,
                'center': (x + w // 2, y + h // 2)
            })
            
    return annotated_img, detected_stones
