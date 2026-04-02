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
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "faces")
CSV_PATH = os.path.join(REPO_ROOT, "data", "static_mesh_dataset.csv")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 6 emotions 
EMOTIONS = {"angry": 0, "disgust": 1, "fear": 2, "happy": 3, "neutral": 4, "sad": 5}

def calc_distance(p1, p2, w, h):
    # multiply by width and height to fix Webcam vs Dataset Aspect Ratio stretching
    #calc 3d euclidean distance between 2 mediapipe landmarks
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2 + ((p1.z - p2.z)*w)**2)

def extract_features(landmarks, w, h):
    iod = calc_distance(landmarks[133], landmarks[362], w, h) + 1e-6    #1e-6 is 0.00001 (epsilon) and prevents division by 0
    return [
        # mouth aspect ration(MAR): openness of the mouth
        # Top inner lip: 13, Bottom inner lip: 14, Left corner: 78, Right corner: 308
        calc_distance(landmarks[78], landmarks[308], w, h) / iod,   # Mouth W
        calc_distance(landmarks[13], landmarks[14], w, h) / iod,    # Mouth H
        # left eye aspect ratio (EAR): openess of the eye
        # Top: 159, Bottom: 145, Inner: 133, Outer: 33
        calc_distance(landmarks[33], landmarks[133], w, h) / iod,   # L Eye W
        calc_distance(landmarks[159], landmarks[145], w, h) / iod,  # L Eye H
        # right eye aspect ration (EAR)
        # Top: 386, Bottom: 374, Inner: 362, Outer: 263
        calc_distance(landmarks[362], landmarks[263], w, h) / iod,  # R Eye W
        calc_distance(landmarks[386], landmarks[374], w, h) / iod,  # R Eye H
        #  eyebrow arch: distance from eye center to eyebrow
        calc_distance(landmarks[105], landmarks[159], w, h) / iod,  # L Brow Inner
        calc_distance(landmarks[334], landmarks[386], w, h) / iod,  # R Brow Inner

        calc_distance(landmarks[107], landmarks[33], w, h) / iod,   # L Brow Outer
        calc_distance(landmarks[336], landmarks[263], w, h) / iod   # R Brow Outer
    ]


#main app
if __name__ == "__main__":
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["label"] + [f"feature_{i}" for i in range(10)])

        for emo_name, label_idx in EMOTIONS.items():
            emo_dir = os.path.join(DATA_DIR, emo_name)
            if not os.path.exists(emo_dir): continue
            
            for img_name in tqdm(os.listdir(emo_dir), desc=f"Processing {emo_name}"):
                image = cv2.imread(os.path.join(emo_dir, img_name))
                if image is None: continue
                
                h, w, _ = image.shape
                
                # MediaPipe Tasks API requires a specific mp.Image object
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #required for mediapipe (opencv has BGR by default), we need to transform to RGB
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                
                results = detector.detect(mp_image)
                if len(results.face_landmarks) > 0: # if a face is detected:
                    feats = extract_features(results.face_landmarks[0], w, h) #face_landmarks is a list of faces; bcz we told mediapipe to only look for 1 face, the list only has one item in it; [0] grabs that face
                    writer.writerow([label_idx] + feats)
    print("Math Extraction done")