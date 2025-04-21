import os
import cv2
from gfpgan import GFPGANer

model = GFPGANer(model_path="Attendance/GFPGANv1.4.pth",
                 upscale=4, arch='clean', channel_multiplier=2)

input_dir = "Attendance/LN02/cropped_faces"
output_dir = "Attendance/LN02/super_resolved_faces"
os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(input_dir):
    input_path = os.path.join(input_dir, img_name)
    output_path = os.path.join(output_dir, img_name)

    img = cv2.imread(input_path)
    if img is None:
        print(f"Skipping {img_name} (Could not read image)")
        continue

    _, _, output = model.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)

    cv2.imwrite(output_path, output)
    print(f"✅ Processed {img_name}")

print("Super-resolution complete for all images in 'cropped_faces'!")
