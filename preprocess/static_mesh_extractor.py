#static_mesh_extractor.py
import os
import cv2
import csv
import math
from tqdm import tqdm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'face_landmarker.task')


CK_DIR = os.path.join(REPO_ROOT, "data", "raw", "ck") 
NEUTRAL_DIR = os.path.join(REPO_ROOT, "data", "processed", "faces", "neutral")
CSV_PATH = os.path.join(REPO_ROOT, "data", "static_mesh_dataset.csv")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)




CK_EMOTIONS = {"anger": 0, "disgust": 1, "fear": 2, "happy": 3, "sadness": 5}


def calc_distance(p1, p2, w, h):
    
    
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2 + ((p1.z - p2.z)*w)**2)

def extract_features(landmarks, w, h):
    iod = calc_distance(landmarks[133], landmarks[362], w, h) + 1e-6    
    return [
        
        
        calc_distance(landmarks[78], landmarks[308], w, h) / iod,   
        calc_distance(landmarks[13], landmarks[14], w, h) / iod,    
        
        
        calc_distance(landmarks[33], landmarks[133], w, h) / iod,   
        calc_distance(landmarks[159], landmarks[145], w, h) / iod,  
        
        
        calc_distance(landmarks[362], landmarks[263], w, h) / iod,  
        calc_distance(landmarks[386], landmarks[374], w, h) / iod,  
        
        calc_distance(landmarks[105], landmarks[159], w, h) / iod,  
        calc_distance(landmarks[334], landmarks[386], w, h) / iod,  

        calc_distance(landmarks[107], landmarks[33], w, h) / iod,   
        calc_distance(landmarks[336], landmarks[263], w, h) / iod   
    ]


def process_image(img_path, label_idx,writer):
    image= cv2.imread(img_path)
    if image is None:return
    
    # MediaPipe Tasks API requires a specific mp.Image object
    h,w,_ = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #required for mediapipe (opencv has BGR by default), we need to transform to RGB
    mp_image=mp.Image(image_format= mp.ImageFormat.SRGB, data=image_rgb)
    
    results= detector.detect(mp_image)
    if len(results.face_landmarks)>0:
        feats= extract_features(results.face_landmarks[0], w,h) #face_landmarks is a list of faces; bcz we told mediapipe to only look for 1 face, the list only has one item in it; [0] grabs that face
        writer.writerow([label_idx] +feats)
        

if __name__ == "__main__":
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"feature_{i}" for i in range(10)])

        for emo_name, label_idx in CK_EMOTIONS.items():
            emo_dir = os.path.join(CK_DIR, emo_name)
            if not os.path.exists(emo_dir): continue
            
            for img_name in tqdm(os.listdir(emo_dir), desc=f"Processing CK+ {emo_name}"):
                process_image(os.path.join(emo_dir, img_name), label_idx, writer)
                
        if os.path.exists(NEUTRAL_DIR):
            images = os.listdir(NEUTRAL_DIR)
            for img_name in tqdm(images[:120],desc="neutral (processed)"): # 120 so that it is about the same amout of neutral emotions as the rest
                process_image(os.path.join(NEUTRAL_DIR,img_name),4,writer)              

    print("Math Extraction done")