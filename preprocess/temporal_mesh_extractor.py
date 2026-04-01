import os
import cv2
import csv
import math
from collections import deque
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))
MODEL_PATH= os.path.join(REPO_ROOT, 'models', 'face_landmarker.task')
VIDEO_DIR= os.path.join (REPO_ROOT, 'data', 'processed', 'mp4_faces')
CSV_PATH= os.path.join(REPO_ROOT, 'data', 'temporal_mesh_dataset.csv')

base_options= python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

EMOTIONS = {"amgry":0, "fear":1, "happy":2, "neutral":3, "sad":4} # dictionary

def calc_distance(p1,p2):
    #calc 3d euclidean distance between 2 mediapipe landmarks
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 +(p1.z-p2.z)**2) #squared bcz we do not want negative values

def extract_ratios(landmarks):
    #converts 478 raw dots into 5 scale-invariant psysiological datas
    # 1) left eye aspect ratio (EAR): openess of the eye
        # Top: 159, Bottom: 145, Inner: 133, Outer: 33

    eye_l_h = calc_distance(landmarks[159], landmarks[145]) #height
    eye_l_w = calc_distance(landmarks[33], landmarks[133])  #width
    ear_l= eye_l_h/(eye_l_w + 1e-6) # 1e-6 is 0.000001, to avoid division by zero
    
    #2) right eye aspect ration (EAR)
        # Top: 386, Bottom: 374, Inner: 362, Outer: 263

    eye_r_h=calc_distance(landmarks[386], landmarks[374])
    eye_r_w=calc_distance(landmarks[362],landmarks[263])
    ear_r= eye_r_h/(eye_r_w + 1e-6)
    
    #3) mouth aspect ration(MAR): openness of the mouth
        # Top inner lip: 13, Bottom inner lip: 14, Left corner: 78, Right corner: 308

    mouth_h=calc_distance(landmarks[13], landmarks[14])
    mouth_w=calc_distance(landmarks[78], landmarks[308])
    mar= mouth_h/(mouth_w + 1e-6)
    
    # 4) distance between the two eyes
    iod = calc_distance(landmarks[133], landmarks[362]) + 1e-6 #iod is interocular dist
    
    # 5) eyebrow arch: distance from eye center to eyebrow
    brow_l = calc_distance(landmarks[105], landmarks[159]) / iod
    brow_r = calc_distance(landmarks[334], landmarks[386]) / iod
    
    return[ear_l, ear_r, mar, brow_l, brow_r]



# VIDEO PROCESSING & SLIDING WINDOW

def process_videos():
    print("extracting temporal data from: {VIDEO_DIR}")
    
    with open(CSV_PATH, mode='w', newline='') as file:
        writer = csv.writer(file)
        
        header= ["actor_id", "label"]
        for frame_idx in range (1,6):
            header.extend([f"ear_l_f{frame_idx}", f"ear_r_f{frame_idx}", f"mar_f{frame_idx}", f"brow_l_f{frame_idx}", f"brow_r_f{frame_idx}"])
        writer.writerow(header)
        
        for emotion_name, label_idx in EMOTIONS.items():
            folder_path = os.path.join(VIDEO_DIR, emotion_name)
            if not os.path.exists(folder_path): continue
            
            videos = [v for v in os.listdir(folder_path) if v.endswith(".mp4")] #this is just double checking, but normally in the VIDEO_DIR there are only mp4s (from the preprocessing)
            
            for vid_name in tqdm(videos, desc=f"processing {emotion_name}"):
                actor_id = int(vid_name.split('-')[-1].split('.')[0]) # -1 is grab the last item (here with a dash), and 0 is the first item (here with a .)
                # ravdess_01-01-05-01-01-01-23.mp4 -> Actor 23
                # in video files where we do sliding windowing, we do not split in train-test, instead we extract actor id and compare the actors. very important 

                vid_path = os.path.join(folder_path, vid_name)
                cap = cv2.VideoCapture(vid_path) #openCV function that opens a video file from ahrd drive and loads into RAM; cap is short for capture. once it is open, u can do cap.read() to step thru the video frame by frame
                
                window = deque(maxlen=5) #stores the 5 frames we need to process (representing time), deques old items when gets full; 
                frame_counter= 0 #keeps track of which frame we are on
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    #process only every 3rd frame (at 30fps, 5 frames=0.5 sec) -> we do this to speed up time
                    if frame_counter %3 ==0:
                        # MediaPipe Tasks API requires a specific mp.Image object
                        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        mp_image=mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
                        
                        results= detector.detect(mp_image)
                        
                        if len(results.face_landmarks) > 0: #if a face is detected:
                            landmarks = results.face_landmarks[0] #face_landmarks is a list of faces; bcz we told mediapipe to only look for 1 face, the list only has one item in it; [0] grabs that face
                            features = extract_ratios(landmarks)
                            window.append(features)
                            
                            #if window == 5, save to csv
                            if len(window) ==5: 
                                #flatten the list of lists into 1 single list of 25 numbers
                                flattened_features = [item for sublist in window for item in sublist] #list compression; this is a shortcut for a nested for-loop (list of lists)
                                '''equivalent to: 
                                
                                for sublist in window:
                                    for item in sublist:
                                        flattened_features.append(item)
                                '''
                                row = [actor_id, label_idx] + flattened_features
                                writer.writerow(row)
                                
                    frame_counter +=1
                    
                cap.release()
    print(f"Temporal dataset saved to: {CSV_PATH}")
    
#optional
if __name__ == "__main__":
    process_videos()
                                