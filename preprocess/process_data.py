import os
import shutil
from tqdm import tqdm


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

RAW_DIR = os.path.join(REPO_ROOT, "data", "raw")
DEST_DIR = os.path.join(REPO_ROOT, "data", "processed")
os.makedirs(DEST_DIR, exist_ok=True) #create if doensnt exist

PATH_FER2013 = os.path.join(RAW_DIR, "fer2013")
PATH_RAFDB   = os.path.join(RAW_DIR, "raf-db")
PATH_RAVDESS = os.path.join(RAW_DIR, "ravdess")
PATH_CREMAD  = os.path.join(RAW_DIR, "cremad")

DEST_FACES = os.path.join(DEST_DIR, "faces")
DEST_AUDIO = os.path.join(DEST_DIR, "audio")

# we drop surprise and calm emotions
CORE_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad"]

# create target directories
for emotion in CORE_EMOTIONS:
    os.makedirs(os.path.join(DEST_FACES, emotion), exist_ok=True)
    os.makedirs(os.path.join(DEST_AUDIO, emotion), exist_ok=True)



# we need to process the datasets such that they all treat the same 6 emotions (angry, disgust, fear, happy, neutral, sad) 
# --> some of the datasets have more emotions (ex. surprise, calm) which we will drop 

def process_fer2013():
    print("\n--- Processing FER-2013 (Faces) ---")
    fer_emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    
    for split in ["train", "test"]:
        split_dir = os.path.join(PATH_FER2013, split)
        if not os.path.exists(split_dir): continue
            
        for emo in fer_emotions:
            emo_dir = os.path.join(split_dir, emo)
            if not os.path.exists(emo_dir): continue
                
            files = os.listdir(emo_dir)
            for file in tqdm(files, desc=f"FER2013 {split}/{emo}"):
                src = os.path.join(emo_dir, file)
                new_name = f"fer2013_{split}_{file}"
                dst = os.path.join(DEST_FACES, emo, new_name)
                shutil.copy2(src, dst)



def process_rafdb():
    print("\n--- Processing RAF-DB (Faces) ---")
    raf_map = {
        "2": "fear", "3": "disgust", "4": "happy", 
        "5": "sad", "6": "angry", "7": "neutral"
    }
    
    dataset_dir = os.path.join(PATH_RAFDB, "DATASET") 

    for split in ["train", "test"]:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir): continue
            
        for folder_num, target_emo in raf_map.items():
            emo_dir = os.path.join(split_dir, folder_num)
            if not os.path.exists(emo_dir): continue
                
            files = os.listdir(emo_dir)
            for file in tqdm(files, desc=f"RAF-DB {split}/{target_emo}"):
                src = os.path.join(emo_dir, file)
                new_name = f"rafdb_{split}_{file}"
                dst = os.path.join(DEST_FACES, target_emo, new_name)
                shutil.copy2(src, dst)

def process_ravdess():
    print("\n--- Processing RAVDESS (Audio) ---")
    ravdess_map = {
        "01": "neutral", "02": "neutral", #calm merged to neutral
        "03": "happy", "04": "sad", 
        "05": "angry", "06": "fear", "07": "disgust"
    }

    for root, _, files in os.walk(PATH_RAVDESS):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 3:
                    emo_code = parts[2]
                    if emo_code in ravdess_map:
                        target_emo = ravdess_map[emo_code]
                        src = os.path.join(root, file)
                        new_name = f"ravdess_{file}"
                        dst = os.path.join(DEST_AUDIO, target_emo, new_name)
                        shutil.copy2(src, dst)

def process_cremad():
    print("\n--- Processing CREMA-D (Audio) ---")
    cremad_map = {
        "ANG": "angry", "DIS": "disgust", "FEA": "fear", 
        "HAP": "happy", "NEU": "neutral", "SAD": "sad"
    }

    for root, _, files in os.walk(PATH_CREMAD):
        wav_files = [f for f in files if f.endswith(".wav")]
        if not wav_files: continue
            
        for file in tqdm(wav_files, desc="CREMA-D files"):
            parts = file.split("_")
            if len(parts) >= 3:
                emo_code = parts[2]
                if emo_code in cremad_map:
                    target_emo = cremad_map[emo_code]
                    src = os.path.join(root, file)
                    new_name = f"cremad_{file}"
                    dst = os.path.join(DEST_AUDIO, target_emo, new_name)
                    shutil.copy2(src, dst)

if __name__ == "__main__":
    print("Starting data preprocessing...\n")
    
    if not os.path.exists(RAW_DIR) or len(os.listdir(RAW_DIR)) == 0:
        print("Error: data/1_raw/ is empty, run data/download_data.py first.")
        exit()

    process_fer2013()
    process_rafdb()
    process_ravdess()
    process_cremad()
    print("\nPreprocessing done./")