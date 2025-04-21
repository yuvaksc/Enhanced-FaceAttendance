import os
import cv2
import numpy as np
import torch
import pickle
from deepface import DeepFace
import face_recognition
from facenet_pytorch import InceptionResnetV1
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms


def normalize_embedding(embedding, target_size=512): # Padding
    if embedding is None:
        return None
    embedding = np.array(embedding)
    current_size = embedding.shape[0]
    if current_size == target_size:
        return embedding
    elif current_size < target_size:
        pad_width = target_size - current_size
        return np.pad(embedding, (0, pad_width), 'constant')
    else:
        return embedding[:target_size] # If larger, truncate

def fuse_embeddings(emb_list, target_size=512): # Normalize each embedding to target_size and concatenate
    norm_embs = [normalize_embedding(e, target_size) for e in emb_list if e is not None]
    if norm_embs and len(norm_embs) == len(emb_list):
        return np.concatenate(norm_embs) # number_of_models * target_size
    return None


def load_facenet_model():
    model = InceptionResnetV1(pretrained='vggface2').eval()
    return model

def extract_facenet_embedding(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Unable to load image {img_path}")
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (160, 160))
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    tensor = transform(img_resized).unsqueeze(0)  # Shape: [1, 3, 160, 160]
    with torch.no_grad():
        embedding = model(tensor)
    return embedding.cpu().numpy().flatten()  # 512-dim


def sface_embeddings(img_path):
    try:
        embedding = DeepFace.represent(img_path, model_name="SFace", enforce_detection=False)
        return embedding[0]["embedding"] # returns a dict with key 'embedding'
    except Exception as e:
        print(f"SFace: failed on {img_path} with error: {e}")
        return None                  



augmented_dir = "data/augmented_enroll"
embeddings_list = []
labels_list = []

facenet_model = load_facenet_model()

for student_id in os.listdir(augmented_dir):
    student_folder = os.path.join(augmented_dir, student_id)
    if not os.path.isdir(student_folder):
        continue

    for img_name in os.listdir(student_folder):
        img_path = os.path.join(student_folder, img_name)
        
        emb_facenet = extract_facenet_embedding(facenet_model, img_path)
        emb_sface = sface_embeddings(img_path)
        
        fused_emb = fuse_embeddings([emb_facenet, emb_sface], target_size=512) # Fuse the embeddings from all models (each normalized to 512 dimensions)
        
        if fused_emb is not None:
            embeddings_list.append(fused_emb)
            labels_list.append(student_id)
        else:
            print(f"Skipping image {img_path}: incomplete embeddings.")

    print("Embeddings Evaluated for Student: ", student_id)
    

X = np.array(embeddings_list)
y = np.array(labels_list)
np.save("embeddings.npy", X)
np.save("labels.npy", y)
print(f"Extracted embeddings for {len(X)} images.")

# -------------------
# SVM Training
# -------------------
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# svm_classifier = SVC(kernel='linear', probability=True)
# svm_classifier.fit(X_train, y_train)

# y_train_pred = svm_classifier.predict(X_train)
# acc = accuracy_score(y_train, y_train_pred)
# print(f"Training Accuracy: {acc * 100:.2f}%")


# y_pred = svm_classifier.predict(X_val)
# accuracy = accuracy_score(y_val, y_pred)
# print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# # Save the classifier
# with open("svm_classifier.pkl", "wb") as f:
#     pickle.dump(svm_classifier, f)
# print("SVM classifier saved to 'svm_classifier.pkl'.")
