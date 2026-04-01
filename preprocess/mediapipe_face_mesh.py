# FEATURE EXTRACTION , DATA PREPROCESSING -> NOT TRAINING YET

import os
import cv2
import csv
from tqdm import tqdm

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ==========================================
# 1. SETUP MEDIAPIPE TASKS API
# ==========================================

MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'face_landmarker.task'))
# MODEL_PATH = "../models/face_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceLandmarkerOptions(
    base_options=base_options, #tells mediapipe where the weights (the model) are
    output_face_blendshapes=False, #measures for e.g. how widely a jaw is open on a scale of 0 to 1, but this is too much details for us, we turn off to save CPU power
    output_facial_transformation_matrixes=False, #we dont need the 3d rotation matrix of the head
    num_faces=1 #only track the largest face you see
)

# init the detector
detector = vision.FaceLandmarker.create_from_options(options) #loads the options and .task in the PC's RAM

# ==========================================
# 2. DATASET CONFIGURATION
# ==========================================

EMOTIONS = {
    "angry": 0,
    "fear": 1,
    "happy": 2,
    "neutral": 3,
    "sad": 4
}

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(REPO_ROOT, "data", "processed", "faces") #processed faces
CSV_PATH = os.path.join(REPO_ROOT, "data", "face_mesh_dataset.csv") #target directory

def process_images_to_csv(): #builds the CSV header (the columns)
    print(f"Loading images from: {DATA_DIR}")
    
    with open(CSV_PATH, mode='w', newline='') as file: # w is open in write mode, newline prevents from accidentally adding a blank empty row between rows
        writer = csv.writer(file)
        
        # write the header row. The new Task model outputs 478 points
        header = ["label"] #1st column is "label" which will hold the 0-4 emotion integer
        for i in range(478):
            header.extend([f"x{i}", f"y{i}", f"z{i}"])
        writer.writerow(header)

        # loop through the 5 emotion folders
        for emotion_name, label_idx in EMOTIONS.items():
            folder_path = os.path.join(DATA_DIR, emotion_name)
            
            if not os.path.exists(folder_path):
                print(f"⚠️ Skipping {emotion_name}, folder not found at {folder_path}.")
                continue
                
            images = os.listdir(folder_path)
            
            for img_name in tqdm(images, desc=f"Processing {emotion_name}"):
                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path) #loads the img into memory as a 3D numpy array of pixels
                if image is None: continue
                
                # MediaPipe Tasks API requires a specific mp.Image object
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #swap the color channels from BGR(OpenCV default) to RGB (MediaPipe)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb) 
                
                # detect the face mesh: scan the img, find the face, calc the 478 3D coordinates
                detection_result = detector.detect(mp_image)
                
                # if a face was found
                if len(detection_result.face_landmarks) > 0:
                    landmarks = detection_result.face_landmarks[0] #grab the 1st face it found
                    
                    # normalization: find the tip of the nose (landmark 1) to use as origin (0,0,0)
                    nose_x = landmarks[1].x
                    nose_y = landmarks[1].y
                    nose_z = landmarks[1].z
                    # --> in mediapipe face mesh topology, point1 is exactly the tip of the nose
                    
                    
                    
                    row = [label_idx] #build the data row for the current img. Put the emotion int (label_idx) in the 1st col
                    
                    
                    #---DATA NORMALISATION: if a person's face is on the left side of the screen, all their X coordinates might be around 0.2
                    # if they are on the right side, their X coords might be 0.8. If u pass the raw coords. to the AI, it will memorize where
                    # the person is standing, not what emotion they are feeling. By substracting the nose from every point on the face, we shift 
                    #the entire face so the nose is perfectly at the center (0,0,0).---
                    for lm in landmarks:
                        # subtract nose coordinates to make the data translation-invariant
                        norm_x = lm.x - nose_x
                        norm_y = lm.y - nose_y
                        norm_z = lm.z - nose_z
                        row.extend([norm_x, norm_y, norm_z])
                        
                    writer.writerow(row) #saves final row (1 label + 1434 numbers (=478 * 3))

    print(f"\n✅ Extraction complete! Dataset saved to {CSV_PATH}")

if __name__ == "__main__":
    process_images_to_csv()