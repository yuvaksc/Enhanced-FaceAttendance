import cv2
import os
import numpy as np

input_folder = "data/enroll"
output_folder = "data/processed_enroll"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def denoise_image(image): # Non Local Denoising
    return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

for student_id in os.listdir(input_folder):
    student_path = os.path.join(input_folder, student_id)
    output_student_path = os.path.join(output_folder, student_id)

    if not os.path.exists(output_student_path):
        os.makedirs(output_student_path)

    for img_name in os.listdir(student_path):
        img_path = os.path.join(student_path, img_name)
        output_img_path = os.path.join(output_student_path, img_name)

        image = cv2.imread(img_path)

        if image is None:
            continue 

        denoised = denoise_image(image)
        sharpened = unsharp_mask(denoised)

        cv2.imwrite(output_img_path, sharpened)

print("Processing complete! Enhanced images saved in 'data/processed_enroll'.")
