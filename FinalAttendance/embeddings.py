import os
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from deepface import DeepFace
from torchvision import transforms
import pickle


def normalize_embedding(embedding, target_size=512):
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
        return embedding[:target_size]

def fuse_embeddings(emb_list, target_size=512):
    norm_embs = [normalize_embedding(e, target_size) for e in emb_list if e is not None]
    if norm_embs and len(norm_embs) == len(emb_list):
        return np.concatenate(norm_embs)
    return None


def load_facenet_model():
    return InceptionResnetV1(pretrained='vggface2').eval()

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
        return embedding[0]["embedding"]
    except Exception as e:
        print(f"SFace: failed on {img_path} with error: {e}")
        return None


input_dir = "take_attendance2"
embeddings_list = []
file_names = []

facenet_model = load_facenet_model()

for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    
    emb_facenet = extract_facenet_embedding(facenet_model, img_path)
    emb_sface = sface_embeddings(img_path)
    
    fused_emb = fuse_embeddings([emb_facenet, emb_sface], target_size=512)
    
    if fused_emb is not None:
        embeddings_list.append(fused_emb)
        file_names.append(img_name)
    else:
        print(f"Skipping image {img_name}: incomplete embeddings.")

print(f"✅ Extracted embeddings for {len(embeddings_list)} images.")

embeddings_data = {"embeddings": np.array(embeddings_list), "file_names": file_names}
with open("model/attendance_embeddings2.pkl", "wb") as f:
    pickle.dump(embeddings_data, f)

print("Attendance embeddings saved successfully!")
