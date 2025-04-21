import torch
import cv2
import os
import numpy as np
from torchvision import transforms
from PIL import Image
from dce_model import enhance_net_nopool

model_path = "SessionCapture/zdce.pth"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = enhance_net_nopool().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

def adaptive_merge(original, enhanced, alpha=0.6, beta=0.4, gamma=0):
    # Blend original and enhanced images adaptively using weighted addition.
    # Alpha controls the influence of the original image, beta influences the enhanced.
    return cv2.addWeighted(original, alpha, enhanced, beta, gamma)

input_dir = "denoised_sessions"
output_dir = "denoised_sessions/enhanced_sessions"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

session_path = os.path.join(input_dir, 'LN02')
output_session_path = os.path.join(output_dir, 'LN02')

if not os.path.exists(output_session_path):
    os.makedirs(output_session_path)

for img_name in os.listdir(session_path):
    img_path = os.path.join(session_path, img_name)

    img = Image.open(img_path).convert("RGB")
    img_resized = img.resize((768, 1024), Image.BICUBIC)
    img_tensor = transform(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        _, enhanced_img, _ = model(img_tensor)
    
    enhanced_img = enhanced_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    enhanced_img = (enhanced_img * 255).clip(0, 255).astype(np.uint8)
    
    enhanced_pil = Image.fromarray(enhanced_img)
    upscaled_image = enhanced_pil.resize(img.size, Image.BICUBIC)
    upscaled_np = np.array(upscaled_image)
    
    original_np = np.array(img)
    
    final_image = adaptive_merge(original_np, upscaled_np, alpha=0.7, beta=0.3) # Apply adaptive contrast correction
    
    output_img_path = os.path.join(output_session_path, img_name)
    cv2.imwrite(output_img_path, cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))

print("Low-light enhancement completed! ✅")