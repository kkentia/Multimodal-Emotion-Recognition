import os
import shutil
import kagglehub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw")

os.makedirs(RAW_DIR, exist_ok=True)

DATASETS = {
    "raf-db": "shuvoalok/raf-db-dataset",
    "fer2013": "msambare/fer2013",
    "ravdess": "orvile/ravdess-dataset",
    "cremad": "ejlok1/cremad",
    "ck": "davilsena/ckdataset"
}

def download_and_move():
    print(f"Files will be saved to: {RAW_DIR}\n")
    
    for folder_name, kaggle_path in DATASETS.items():
        target_path = os.path.join(RAW_DIR, folder_name)
        
        # skip if already downloaded
        if os.path.exists(target_path) and len(os.listdir(target_path)) > 0:
            continue
            
        print(f" Downloading {folder_name}...")
        # kagglehub downloads to a cache and returns the cache path
        cache_path = kagglehub.dataset_download(kaggle_path)
        
        print(f"Copying {folder_name} to local repository...")
        shutil.copytree(cache_path, target_path, dirs_exist_ok=True)
        print(f"{folder_name} successfully copied.\n")

if __name__ == "__main__":
    download_and_move()
    print("All raw datasets are in data/raw/")