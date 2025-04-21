import os
import cv2
import albumentations as A
import numpy as np

input_dir = "data/super_resolved_enroll"
output_dir = "data/augmented_enroll"
os.makedirs(output_dir, exist_ok=True)

# Define augmentation pipeline using Albumentations
augmentation_pipeline = A.Compose([
    A.Rotate(limit=15, p=0.8),            # Random rotation ±15 degrees
    A.RandomScale(scale_limit=0.1, p=0.5),  # Random scaling between 0.9x and 1.1x
    A.HorizontalFlip(p=0.5),              # 50% chance to flip horizontally
    A.RandomBrightnessContrast(p=0.5),    # Adjust brightness/contrast
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7)
])

def augment_image(image, num_augmentations=100):
    augmented_images = []
    for i in range(num_augmentations):
        augmented = augmentation_pipeline(image=image)
        augmented_images.append(augmented['image'])
    return augmented_images

for student_id in os.listdir(input_dir):
    student_input_path = os.path.join(input_dir, student_id)
    student_output_path = os.path.join(output_dir, student_id)
    os.makedirs(student_output_path, exist_ok=True)
    
    for img_name in os.listdir(student_input_path):
        img_path = os.path.join(student_input_path, img_name)
        image = cv2.imread(img_path)
        if image is None:
            print(f"Unable to read {img_path}")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_imgs = augment_image(image_rgb, num_augmentations=100)
        
        base_name, ext = os.path.splitext(img_name)
        for idx, aug_img in enumerate(augmented_imgs):
            aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            save_filename = f"{base_name}_aug_{idx+1}{ext}"
            save_path = os.path.join(student_output_path, save_filename)
            cv2.imwrite(save_path, aug_img_bgr)
            print(f"Saved {save_filename} for student {student_id}")

print("✅ Data augmentation complete for all registered face images!")
