import os
import shutil
from tqdm import tqdm

# ==========================================
# 1. SETUP PATHS
# ==========================================
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Assuming your raw RAVDESS dataset was downloaded here via your download.py script
RAW_RAVDESS_DIR = os.path.join(REPO_ROOT, "data", "raw", "ravdess")
DEST_DIR = os.path.join(REPO_ROOT, "data", "processed", "mp4_faces")
os.makedirs(DEST_DIR, exist_ok=True)

# ==========================================
# 2. RAVDESS EMOTION MAPPING
# ==========================================
# RAVDESS filename format: Modality-VocalChannel-Emotion-Intensity-Statement-Repetition-Actor.mp4
# Example: 02-01-05-01-01-01-01.mp4 (The 3rd part '05' is Angry)

EMOTION_MAP = {
    "01": "neutral",
    "02": "neutral", # merging calm into neutral
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fear",
    "07": "disgust"  
}

def organize_ravdess_videos():
    print(f"Scanning raw RAVDESS directory: {RAW_RAVDESS_DIR}")
    
    # create target directories
    for emo_name in set(EMOTION_MAP.values()):
        os.makedirs(os.path.join(DEST_DIR, emo_name), exist_ok=True)
        
    mp4_files =[]
    
    # Walk through the raw directory to find all .mp4 files
    for root, dirs, files in os.walk(RAW_RAVDESS_DIR):
        for file in files:
            if file.endswith(".mp4"):
                mp4_files.append((root, file))
                
    if not mp4_files:
        print("❌ No .mp4 files found! Make sure you downloaded the RAVDESS Video dataset, not just Audio.")
        return

    print(f"Found {len(mp4_files)} video files. Sorting into {DEST_DIR}...")

    # Copy and sort files
    for root, file in tqdm(mp4_files, desc="Copying Videos"):
        parts = file.split("-")
        
        # Ensure it's a valid RAVDESS filename
        if len(parts) >= 3:
            emotion_code = parts[2]
            
            # If the emotion is in our map (ignoring surprise)
            if emotion_code in EMOTION_MAP:
                target_emotion = EMOTION_MAP[emotion_code]
                
                src_path = os.path.join(root, file)
                
                # Prepend 'ravdess_' to prevent any naming collisions if you add more datasets later
                new_filename = f"ravdess_{file}"
                dst_path = os.path.join(DEST_DIR, target_emotion, new_filename)
                
                shutil.copy2(src_path, dst_path)

    print("\n✅ Video sorting complete!")

if __name__ == "__main__":
    organize_ravdess_videos()