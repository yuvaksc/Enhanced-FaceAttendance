# Enhanced-FaceAttendance

# Face Attendance System 📸🎓

An intelligent, real-world-ready face biometric system to automate classroom attendance using robust image enhancement, face detection, and recognition techniques.

## 🔍 Overview

This project presents a robust **Face Attendance System** designed to operate effectively in **real classroom conditions** with varying lighting, occlusion, and non-frontal face poses. It combines state-of-the-art computer vision techniques for preprocessing, detection, and recognition to ensure **high accuracy** even in challenging environments.

---

## 🧠 Key Features

- **Face Preprocessing Pipeline**: Denoising, unsharp masking, region-based sharpening using dlib.
- **Resolution Enhancement**: Leveraging **GFPGAN** for restoring low-res or degraded face images.
- **Face Augmentation**: Multiple transformations per student for robust embedding generation.
- **Dual Embedding Extraction**: Using **FaceNet** and **SFace** for better identity representation.
- **SVM Classification**: Trained per student to identify roll numbers accurately.
- **YOLO Face Detection with Tiling**: Ensures small, distant faces in full classroom images are not missed.
- **Global NMS**: Prevents duplicate detections across image tiles.
- **Low-Light Enhancement**: Enhances classroom images using **Zero-DCE**.

---

## 📁 Pipeline Breakdown

### Phase 1: **Student Registration (Training Phase)**

1. **Capture Student Face Images** (Frontal images)
2. **Preprocessing**:
   - `fastNlMeansDenoisingColored`
   - Unsharp Masking
   - Region-based sharpening using `dlib`
3. **Enhancement**: GFPGAN for face resolution restoration
4. **Augmentation**: Increase data variety using flips, crops, etc.
5. **Embedding Extraction**:
   - **FaceNet**
   - **SFace**
6. **Training**: Train an **SVM** classifier on embeddings → Maps faces to **Roll Numbers**

---

### Phase 2: **Classroom Session (Testing Phase)**

1. **Input**: Classroom images (frames from video)
2. **Preprocessing**:
   - Gaussian Blur + Bilateral Filter
   - `fastNlMeansDenoisingColored`
3. **Lighting Enhancement**: Apply **ZeroDCE**
4. **Face Detection**:
   - YOLO with tiling
   - Global Non-Maximum Suppression (NMS)
5. **Enhancement**: GFPGAN
6. **Embedding Extraction**: SFace + FaceNet
7. **Prediction**: Use trained SVM to identify roll numbers
8. **Attendance Log**: Save recognized roll numbers

---

## ⚙️ Tech Stack

- Python
- OpenCV
- Dlib
- GFPGAN
- YOLOv5
- FaceNet (via Keras/TF)
- InsightFace (SFace)
- Zero-DCE (Low-light enhancement)
- Scikit-learn (SVM)

---

## 🧪 Performance & Limitations

✅ **Accurate** on real classroom videos  
✅ Works well even in **low-light** or **partially occluded** scenes  
❌ Slight drop in accuracy for **heavily crowded** scenes or **extreme head poses**

---

## 📈 Future Work

- Integrate **temporal tracking** across frames for continuous presence
- Handle **extreme occlusion** via 3D face reconstruction
- Replace SVM with **lightweight deep classifier**
- Mobile/Web interface for real-time monitoring

---

## 📝 Authors

- Yuva Komara

---

## 📌 License

This project is open-source under the [MIT License](LICENSE).

