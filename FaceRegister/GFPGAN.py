import os
import cv2
import torch
from gfpgan import GFPGANer

model = GFPGANer(model_path="FaceRegister/GFPGANv1.4.pth",
                 upscale=4, arch='clean', channel_multiplier=2)

input_dir = "data/sharpened_enroll"
output_dir = "data/super_resolved_enroll"
os.makedirs(output_dir, exist_ok=True)

for student_id in os.listdir(input_dir):
    student_folder = os.path.join(input_dir, student_id)
    output_student_folder = os.path.join(output_dir, student_id)
    os.makedirs(output_student_folder, exist_ok=True)

    for img_name in os.listdir(student_folder):
        input_path = os.path.join(student_folder, img_name)
        output_path = os.path.join(output_student_folder, img_name)

        img = cv2.imread(input_path)
        _, _, output = model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

        cv2.imwrite(output_path, output)
        print(f"✅ Processed {img_name}")

print("🎉 Super-resolution complete for all students!")
