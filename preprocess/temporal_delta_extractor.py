import os
import cv2
import csv
import math
import mediapipe as mp
from tqdm import tqdm
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(REPO_ROOT, 'models', 'face_landmarker.task')
VIDEO_DIR = os.path.join(REPO_ROOT, "data", "processed", "mp4_faces")
CSV_PATH = os.path.join(REPO_ROOT, "data", "convlstm_dataset.csv")

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

EMOTIONS = {"angry": 0, "fear": 1, "happy": 2, "neutral": 3, "sad": 4}
SEQ_LENGTH = 10 # 10 frames of temporal motion

def calc_dist(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2 + ((p1.z - p2.z)*w)**2)

def extract_features(landmarks, w, h):
    iod = calc_dist(landmarks[133], landmarks[362], w, h) + 1e-6
    return [
        calc_dist(landmarks[78], landmarks[308], w, h) / iod,
        calc_dist(landmarks[13], landmarks[14], w, h) / iod,
        calc_dist(landmarks[33], landmarks[133], w, h) / iod,
        calc_dist(landmarks[159], landmarks[145], w, h) / iod,
        calc_dist(landmarks[362], landmarks[263], w, h) / iod,
        calc_dist(landmarks[386], landmarks[374], w, h) / iod,
        calc_dist(landmarks[105], landmarks[159], w, h) / iod,
        calc_dist(landmarks[107], landmarks[33], w, h) / iod,
        calc_dist(landmarks[334], landmarks[386], w, h) / iod,
        calc_dist(landmarks[336], landmarks[263], w, h) / iod
    ]

def process_videos():
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["actor_id", "label"] +[f"f{f}_feat{i}" for f in range(SEQ_LENGTH) for i in range(10)]
        writer.writerow(header)

        for emo_name, label_idx in EMOTIONS.items():
            emo_dir = os.path.join(VIDEO_DIR, emo_name)
            if not os.path.exists(emo_dir): continue
            
            videos =[v for v in os.listdir(emo_dir) if v.endswith(".mp4")]
            
            for vid_name in tqdm(videos, desc=f"Processing {emo_name}"):
                # Extract RAVDESS Actor ID: ravdess_01-01-05-01-01-01-23.mp4 -> 23
                try: actor_id = int(vid_name.split('-')[-1].split('.')[0])
                except: continue
                
                cap = cv2.VideoCapture(os.path.join(emo_dir, vid_name))
                window = deque(maxlen=SEQ_LENGTH)
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Process every 2nd frame (10 frames = ~0.6 seconds of smooth motion)
                    if frame_count % 2 == 0:
                        h, w, _ = frame.shape
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        res = detector.detect(mp_img)
                        
                        if len(res.face_landmarks) > 0:
                            feats = extract_features(res.face_landmarks[0], w, h)
                            window.append(feats)
                            if len(window) == SEQ_LENGTH:
                                flattened = [val for sublist in window for val in sublist]
                                writer.writerow([actor_id, label_idx] + flattened)
                        else:
                            window.clear()
                    frame_count += 1
                cap.release()

if __name__ == "__main__":
    process_videos()