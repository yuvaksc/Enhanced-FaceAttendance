import cv2
import os

input_base = "data/sessions/"
output_base = "denoised_sessions/"

if not os.path.exists(output_base):
    os.makedirs(output_base)

for session in os.listdir(input_base):
    session_path = os.path.join(input_base, session)
    
    if os.path.isdir(session_path): 
        output_session_path = os.path.join(output_base, session)
        os.makedirs(output_session_path, exist_ok=True)
        
        for image_name in os.listdir(session_path):
            image_path = os.path.join(session_path, image_name)
            
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                print(f"Processing: {image_path}")
                
                image = cv2.imread(image_path)
                
                gaussian_blur = cv2.GaussianBlur(image, (5, 5), 1.5)
                bilateral = cv2.bilateralFilter(gaussian_blur, d=9, sigmaColor=75, sigmaSpace=75)
                nlm_denoised = cv2.fastNlMeansDenoisingColored(bilateral, None, 10, 10, 7, 21)
                
                cv2.imwrite(os.path.join(output_session_path, f"{image_name}"), nlm_denoised)

print("✅ Denoising complete for all session images!")
