import os
import cv2
import csv
import math
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'face_landmarker.task')
CK_DIR = os.path.join(REPO_ROOT, "data", "raw", "ck") # Pointing to your CK+ static images
CSV_PATH = os.path.join(REPO_ROOT, "data", "static_mesh_dataset.csv")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# Your exact CK+ Emotions
EMOTIONS = {"anger": 0, "fear": 1, "happy": 2, "contempt": 3, "sadness": 4}

def calc_distance(p1, p2):
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

def extract_normalized_features(landmarks):
    # 1. Inter-ocular distance (Distance between eyes to use as our base ruler)
    iod = calc_distance(landmarks[133], landmarks[362]) + 1e-6
    
    features =[]
    # Mouth width & height
    features.append(calc_distance(landmarks[78], landmarks[308]) / iod)
    features.append(calc_distance(landmarks[13], landmarks[14]) / iod)
    # Left eye width & height
    features.append(calc_distance(landmarks[33], landmarks[133]) / iod)
    features.append(calc_distance(landmarks[159], landmarks[145]) / iod)
    # Right eye width & height
    features.append(calc_distance(landmarks[362], landmarks[263]) / iod)
    features.append(calc_distance(landmarks[386], landmarks[374]) / iod)
    # Left eyebrow distances (inner, mid, outer to eye)
    features.append(calc_distance(landmarks[105], landmarks[159]) / iod)
    features.append(calc_distance(landmarks[107], landmarks[33]) / iod)
    # Right eyebrow distances
    features.append(calc_distance(landmarks[334], landmarks[386]) / iod)
    features.append(calc_distance(landmarks[336], landmarks[263]) / iod)
    
    return features

def process_ck_static():
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        # Header: 10 normalized features
        header = ["label"] +[f"feature_{i}" for i in range(10)]
        writer.writerow(header)

        for emotion_name, label_idx in EMOTIONS.items():
            emo_dir = os.path.join(CK_DIR, emotion_name)
            if not os.path.exists(emo_dir): continue
            
            images = os.listdir(emo_dir)
            for img_name in tqdm(images, desc=f"Processing {emotion_name}"):
                img_path = os.path.join(emo_dir, img_name)
                
                # Check if it is an image file
                if not os.path.isfile(img_path): continue
                    
                image = cv2.imread(img_path)
                if image is None: continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                results = detector.detect(mp_image)
                if len(results.face_landmarks) > 0:
                    feats = extract_normalized_features(results.face_landmarks[0])
                    writer.writerow([label_idx] + feats)

if __name__ == "__main__":
    process_ck_static()