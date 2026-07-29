import os
import cv2
import numpy as np
import random
from PIL import Image

def generate_ct_scan(has_stone=False, seed=None):
    """
    Generates a realistic synthetic coronal abdominal CT scan image.
    Simulates renal parenchyma, spinal vertebrae, soft tissue density, and optional renal calculi.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    size = (224, 224)
    img = np.zeros(size, dtype=np.float32)

    # 1. Background soft tissue noise & abdominal body contour (ellipse)
    y, x = np.ogrid[:size[0], :size[1]]
    center_y, center_x = 112, 112
    mask_body = ((x - center_x) ** 2) / (85 ** 2) + ((y - center_y) ** 2) / (95 ** 2) <= 1.0
    img[mask_body] = np.random.normal(50, 8, size=np.sum(mask_body))

    # 2. Spine column (high intensity bone in center bottom)
    spine_mask = ((x - 112) ** 2) / (18 ** 2) + ((y - 170) ** 2) / (25 ** 2) <= 1.0
    img[spine_mask] = np.random.normal(180, 15, size=np.sum(spine_mask))

    # 3. Left and Right Kidneys (bean-shaped ellipses)
    # Left kidney (viewer's left)
    left_k_mask = ((x - 65) ** 2) / (28 ** 2) + ((y - 110) ** 2) / (45 ** 2) <= 1.0
    img[left_k_mask] = np.random.normal(90, 10, size=np.sum(left_k_mask))

    # Right kidney (viewer's right)
    right_k_mask = ((x - 159) ** 2) / (28 ** 2) + ((y - 110) ** 2) / (45 ** 2) <= 1.0
    img[right_k_mask] = np.random.normal(90, 10, size=np.sum(right_k_mask))

    # Renal pelvis (inner slightly darker region)
    left_p_mask = ((x - 65) ** 2) / (12 ** 2) + ((y - 110) ** 2) / (20 ** 2) <= 1.0
    img[left_p_mask] = np.random.normal(65, 8, size=np.sum(left_p_mask))

    right_p_mask = ((x - 159) ** 2) / (12 ** 2) + ((y - 110) ** 2) / (20 ** 2) <= 1.0
    img[right_p_mask] = np.random.normal(65, 8, size=np.sum(right_p_mask))

    # 4. Add Kidney Stone if has_stone is True
    if has_stone:
        # Place stone randomly in left or right kidney
        side = random.choice(['left', 'right'])
        if side == 'left':
            stone_x = random.randint(55, 75)
            stone_y = random.randint(95, 125)
        else:
            stone_x = random.randint(149, 169)
            stone_y = random.randint(95, 125)

        stone_radius = random.randint(4, 9)
        stone_mask = ((x - stone_x) ** 2) + ((y - stone_y) ** 2) <= (stone_radius ** 2)
        
        # High HU density for calcification (very bright white, 220-255)
        img[stone_mask] = np.random.normal(240, 10, size=np.sum(stone_mask))
        
        # Acoustic shadowing behind stone
        shadow_mask = (x >= stone_x - 3) & (x <= stone_x + 3) & (y > stone_y + stone_radius) & (y < stone_y + 35) & left_k_mask if side == 'left' else (x >= stone_x - 3) & (x <= stone_x + 3) & (y > stone_y + stone_radius) & (y < stone_y + 35) & right_k_mask
        img[shadow_mask] = np.maximum(0, img[shadow_mask] - 40)

    # 5. Add CT scan speckle noise & Gaussian blur smoothing
    img = cv2.GaussianBlur(img, (3, 3), 0)
    noise = np.random.normal(0, 5, size).astype(np.float32)
    img = img + noise
    
    # Clip pixel intensities to [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)

    # Convert to 3-channel RGB for standard CNN processing
    rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return rgb_img

def create_dataset(base_dir="dataset", train_count=100, test_count=25):
    """Creates directory structure and generates dataset images."""
    categories = ['Stone', 'Normal']
    splits = {'train': train_count, 'test': test_count}

    for split, count in splits.items():
        for category in categories:
            dir_path = os.path.join(base_dir, split, category)
            os.makedirs(dir_path, exist_ok=True)
            
            has_stone = (category == 'Stone')
            print(f"Generating {count} images for {split}/{category}...")
            
            for i in range(count):
                seed = hash((split, category, i)) % (2**32)
                img = generate_ct_scan(has_stone=has_stone, seed=seed)
                file_name = f"{category.lower()}_{i+1:04d}.jpg"
                save_path = os.path.join(dir_path, file_name)
                Image.fromarray(img).save(save_path)

    print("Dataset generation complete!")

if __name__ == "__main__":
    base_directory = os.path.abspath("dataset")
    create_dataset(base_directory, train_count=100, test_count=25)
