
import os
import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Points to the root of her project
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Her exact input video directory based on the screenshot
VIDEO_DIR = os.path.join(REPO_ROOT, "data", "processed", "mp4_faces")

# Where the generated black-background images will be saved for SqueezeNet
OUTPUT_DIR = os.path.join(REPO_ROOT, "data", "mesh_images")

# NOTICE: We are explicitly skipping 'disgust' and 'neutral' to boost accuracy!
# We also updated "fearful" to "fear" to perfectly match her folder name.
EMOTIONS = ["angry", "fear", "happy", "sad"] 

# Initialize MediaPipe Face Mesh tools
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def process_videos_to_images():
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
        
        for emo_name in EMOTIONS:
            emo_vid_dir = os.path.join(VIDEO_DIR, emo_name)
            emo_out_dir = os.path.join(OUTPUT_DIR, emo_name)
            
            # If the source folder exists, create the corresponding output folder
            if os.path.exists(emo_vid_dir): 
                os.makedirs(emo_out_dir, exist_ok=True)
            else:
                print(f"Folder not found: {emo_vid_dir}")
                continue
                
            videos = [v for v in os.listdir(emo_vid_dir) if v.endswith(".mp4")]
            
            for vid_name in tqdm(videos, desc=f"Processing {emo_name}"):
                cap = cv2.VideoCapture(os.path.join(emo_vid_dir, vid_name))
                frame_count = 0
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    # Process every 5th frame to avoid having too many identical images
                    if frame_count % 5 == 0:
                        # Convert BGR (OpenCV default) to RGB (MediaPipe default)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = face_mesh.process(rgb_frame)
                        
                        if results.multi_face_landmarks:
                            for face_landmarks in results.multi_face_landmarks:
                                # 1. Create a pure black image matching the video's dimensions
                                black_bg = np.zeros((frame.shape[0], frame.shape[1], 3), dtype=np.uint8)
                                
                                # 2. Draw the MediaPipe "Spiderweb" in white
                                mp_drawing.draw_landmarks(
                                    image=black_bg,
                                    landmark_list=face_landmarks,
                                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                                    landmark_drawing_spec=None,
                                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                                
                                # 3. Resize to exactly 224x224 (SqueezeNet's required input size)
                                final_image = cv2.resize(black_bg, (224, 224))
                                
                                # 4. Save the image (e.g., ravdess_01-01-07..._frame_10.png)
                                out_filename = f"{vid_name.split('.')[0]}_frame_{frame_count}.png"
                                cv2.imwrite(os.path.join(emo_out_dir, out_filename), final_image)
                    
                    frame_count += 1
                cap.release()

if __name__ == "__main__":
    print("Starting Data Extraction...")
    process_videos_to_images()
    print("Done! Images are ready for SqueezeNet.")

