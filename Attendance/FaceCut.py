import cv2
import os
import numpy as np
import torch
import math
from ultralytics import YOLO

# Non-Maximum Suppression
def global_nms(boxes, confs, iou_threshold=0.35):  # Reduced IoU threshold
    if not boxes:
        return []
    
    boxes = np.array(boxes)
    confs = np.array(confs)
    
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = confs.argsort()[::-1]  # Sort by confidence (descending)
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return boxes[keep].astype(int).tolist()

def tiled_yolo_face_detection(image_path, model, tile_size=512, overlap=384, conf_threshold=0.9, iou_threshold=0.25):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return []
    
    h, w, _ = img.shape
    step_x, step_y = tile_size - overlap, tile_size - overlap
    tiles_x, tiles_y = math.ceil((w - overlap) / step_x), math.ceil((h - overlap) / step_y)
    
    all_boxes, all_confs = [], []
    
    for ty in range(tiles_y):
        for tx in range(tiles_x):
            offset_x, offset_y = tx * step_x, ty * step_y
            tile_x2, tile_y2 = min(offset_x + tile_size, w), min(offset_y + tile_size, h)
            tile = img[offset_y:tile_y2, offset_x:tile_x2]
            
            results = model.predict(source=tile, conf=conf_threshold, iou=iou_threshold, verbose=False, imgsz=tile_size)
            
            if len(results) > 0 and results[0].boxes is not None:
                dets = results[0].boxes.xyxy.cpu().numpy() 
                confidences = results[0].boxes.conf.cpu().numpy() 

                for (x1, y1, x2, y2), conf in zip(dets, confidences):
                    gx1, gy1, gx2, gy2 = x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
                    all_boxes.append([gx1, gy1, gx2, gy2])
                    all_confs.append(conf)
    
    final_boxes = global_nms(all_boxes, all_confs, iou_threshold=iou_threshold)
    return final_boxes

def crop_faces(image_path, save_dir, model, count):
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
    
    boxes = tiled_yolo_face_detection(image_path, model)
    
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        print(f"Face {i}: ({x1}, {y1}, {x2}, {y2})")
        # cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        face = img[y1:y2, x1:x2]
        if face.size > 0:
            face_path = os.path.join(save_dir, f"face_{count}.jpg")
            count += 1
            cv2.imwrite(face_path, face)
            print(f"Saved: {face_path}")
    return count

def process_classroom_frames(root_dir):
    model = YOLO("Attendance/yolov8n.pt")  
    count = 0
    for filename in os.listdir(root_dir):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            image_path = os.path.join(root_dir, filename)
            count = crop_faces(image_path, "Attendance/LN02/cropped_faces", model, count)
    
    print("Processing complete!")

process_classroom_frames("denoised_sessions/enhanced_sessions/LN02")
