import cv2
import os
import dlib
import numpy as np

# Downgraded torchvision to .18.1

input_folder = "data/processed_enroll"  
output_folder = "data/sharpened_enroll"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# dlib's face detector, shape predictor 
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("FaceRegister/shape_predictor_68_face_landmarks.dat")

def sharpen_region(image, region, strength=2.0): # localized sharpening
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [region], 255)

    blurred = cv2.GaussianBlur(image, (0, 0), 2)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    
    return np.where(mask[:, :, None] > 0, sharpened, image) # Blends only sharpened regions

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

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            eyes = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 48)])
            nose = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(27, 36)])
            mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(48, 68)])

            # ear regions (Extend from jawline)
            left_ear = np.array([(landmarks.part(n).x - 15, landmarks.part(n).y) for n in range(0, 9)])
            right_ear = np.array([(landmarks.part(n).x + 15, landmarks.part(n).y) for n in range(8, 17)])

            # hairline (Forehead region from top facial landmarks)
            hairline = np.array([(landmarks.part(n).x, landmarks.part(n).y - 20) for n in range(17, 27)])

            image = sharpen_region(image, eyes, strength=2.5)
            image = sharpen_region(image, nose, strength=2.0)
            image = sharpen_region(image, mouth, strength=2.0)
            image = sharpen_region(image, left_ear, strength=1.8)
            image = sharpen_region(image, right_ear, strength=1.8)
            image = sharpen_region(image, hairline, strength=1.5)

        cv2.imwrite(output_img_path, image)

print("Facial feature sharpening complete! Images saved in 'data/sharpened_enroll'.")
