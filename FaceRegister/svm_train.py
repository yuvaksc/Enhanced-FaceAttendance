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

X = np.load('embeddings.npy')
y = np.load('labels.npy')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

svm_classifier = SVC(kernel="rbf", gamma="scale")  # RBF kernel for better generalization
svm_classifier.fit(X_train, y_train)                

y_train_pred = svm_classifier.predict(X_train)              
acc = accuracy_score(y_train, y_train_pred)                             
print(f"Training Accuracy: {acc * 100:.2f}%")                                   


y_pred = svm_classifier.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

with open("svm_classifier1.pkl", "wb") as f:
    pickle.dump(svm_classifier, f)
print("SVM classifier saved to 'svm_classifier.pkl'.")
