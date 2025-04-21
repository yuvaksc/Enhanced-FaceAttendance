import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

# Load saved embeddings
with open("model/attendance_embeddings.pkl", "rb") as f:
    data = pickle.load(f)

embeddings = data["embeddings"]
file_names = data["file_names"]

# Load trained SVM model
svm_path = "model/svm_classifier.pkl"
if not os.path.exists(svm_path):
    raise FileNotFoundError("SVM model not found! Ensure svm_classifier.pkl is in the 'model' folder.")

with open(svm_path, "rb") as f:
    svm_model = pickle.load(f)

# Predict labels
predictions = svm_model.predict(embeddings)

# Get unique names
unique_names = sorted(set(predictions))


print("🎯 **Attendance Taken For:**")
for name in unique_names:
    enroll_dir = os.path.join("data/enroll", name)
    enroll_images = os.listdir(enroll_dir)
    if enroll_images:
        enroll_img_path = os.path.join(enroll_dir, enroll_images[0])
        enroll_img = cv2.imread(enroll_img_path)
        enroll_img = cv2.cvtColor(enroll_img, cv2.COLOR_BGR2RGB)
    else:
        print("Error")
    print(f"✅ {name}")

    plt.imshow(enroll_img)
    plt.show()

print(f"\n🎉 Attendance recorded for {len(unique_names)} unique individuals!")















