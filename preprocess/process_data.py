# this file structures the raw datasets and organizes them by audio/video and emotions
    
import os
import shutil
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
DEST_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(DEST_DIR, exist_ok=True)

PATH_FER2013 = os.path.join(RAW_DIR, "fer2013")
PATH_RAFDB   = os.path.join(RAW_DIR, "raf-db")
PATH_RAVDESS = os.path.join(RAW_DIR, "ravdess")
PATH_CREMAD  = os.path.join(RAW_DIR, "cremad")
PATH_CK      = os.path.join(RAW_DIR, "ck") 

DEST_FACES = os.path.join(DEST_DIR, "faces")
DEST_AUDIO = os.path.join(DEST_DIR, "audio")

CORE_EMOTIONS =["angry", "disgust", "fear", "happy", "neutral", "sad"]

for emotion in CORE_EMOTIONS:
    os.makedirs(os.path.join(DEST_FACES, emotion), exist_ok=True)
    os.makedirs(os.path.join(DEST_AUDIO, emotion), exist_ok=True)

def process_fer2013():
    print("\n--- Processing FER-2013 (Faces) ---")
    for split in ["train", "test"]:
        split_dir = os.path.join(PATH_FER2013, split)
        if not os.path.exists(split_dir): continue
        for emo in CORE_EMOTIONS:
            emo_dir = os.path.join(split_dir, emo)
            if not os.path.exists(emo_dir): continue
            files = os.listdir(emo_dir)
            for file in tqdm(files, desc=f"FER2013 {split}/{emo}"):
                shutil.copy2(os.path.join(emo_dir, file), os.path.join(DEST_FACES, emo, f"fer2013_{split}_{file}"))

def process_rafdb():
    print("\n--- Processing RAF-DB (Faces) ---")
    raf_map = {"2": "fear", "3": "disgust", "4": "happy", "5": "sad", "6": "angry", "7": "neutral"}
    dataset_dir = os.path.join(PATH_RAFDB, "DATASET") 
    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir): continue
        for folder_num, target_emo in raf_map.items():
            emo_dir = os.path.join(split_dir, folder_num)
            if not os.path.exists(emo_dir): continue
            files = os.listdir(emo_dir)
            for file in tqdm(files, desc=f"RAF-DB {split}/{target_emo}"):
                shutil.copy2(os.path.join(emo_dir, file), os.path.join(DEST_FACES, target_emo, f"rafdb_{split}_{file}"))

def process_ck():
    print("\n--- Processing CK+ (Faces) ---")
    ck_map = {"anger": "angry", "disgust": "disgust", "fear": "fear", "happy": "happy", "sadness": "sad"}
    if not os.path.exists(PATH_CK): return
    for folder_name, target_emo in ck_map.items():
        emo_dir = os.path.join(PATH_CK, folder_name)
        if not os.path.exists(emo_dir): continue
        files = os.listdir(emo_dir)
        for file in tqdm(files, desc=f"CK+ {target_emo}"):
            shutil.copy2(os.path.join(emo_dir, file), os.path.join(DEST_FACES, target_emo, f"ck_{file}"))

def process_ravdess():
    print("\n--- Processing RAVDESS (Audio) ---")
    ravdess_map = {"01": "neutral", "02": "neutral", "03": "happy", "04": "sad", "05": "angry", "06": "fear", "07": "disgust"}
    for root, _, files in os.walk(PATH_RAVDESS):
        for file in files:
            if file.endswith(".wav") and len(file.split("-")) >= 3:
                emo_code = file.split("-")[2]
                if emo_code in ravdess_map:
                    shutil.copy2(os.path.join(root, file), os.path.join(DEST_AUDIO, ravdess_map[emo_code], f"ravdess_{file}"))

def process_cremad():
    print("\n--- Processing CREMA-D (Audio) ---")
    cremad_map = {"ANG": "angry", "DIS": "disgust", "FEA": "fear", "HAP": "happy", "NEU": "neutral", "SAD": "sad"}
    for root, _, files in os.walk(PATH_CREMAD):
        for file in tqdm([f for f in files if f.endswith(".wav")], desc="CREMA-D"):
            if len(file.split("_")) >= 3 and file.split("_")[2] in cremad_map:
                shutil.copy2(os.path.join(root, file), os.path.join(DEST_AUDIO, cremad_map[file.split("_")[2]], f"cremad_{file}"))

if __name__ == "__main__":
    print("Starting data preprocessing...\n")
    if not os.path.exists(RAW_DIR) or len(os.listdir(RAW_DIR)) == 0:
        print("Error: data/raw/ is empty.")
        exit()

    process_fer2013()
    process_rafdb()
    process_ck()
    process_ravdess()
    process_cremad()
    print("\nPreprocessing done.")
    
