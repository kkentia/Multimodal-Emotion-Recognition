# Multimodal-Emotion-Recognition
**Project Description:** Combine FER (Facial Emotion Recognition), SER (Speech Emotion Recognition) and NLP (Natural Language Processing) using a late fusion approach.

| Name | Github Handle | Workload
| --- | --- | --- |
| Ana Bog | @kkentia | FER Deep Learning
| Antonia Spörk | @antoniaspoerk | UI + video integration
| Ali Abdel Ghaffar | @AliAgh123 | SER Deep Learning
| Tyler Wilson | @yMakaveli | UI + audio integration

---
**Installation & How to**
 1. clone the repo locally
 2. cd into it, and inside a new virtual env (important!) , do: 'pip install -r requirements.txt' to install dependencies.
 3. run the download_data.py script from inside /data to install the datasets locally. They will appear in the /data/raw folder.
 4. run the process_data.py script from inside /preprocess. Processed data for 6 emotions (audio and faces) will appear in the /data/processed folder.
 5. run the static_mesh_extraction.py to calculate the face geometry relative to itself, creating a .csv file the model will train on.
 6. run the train_static_mesh.py to actually train the mediapipe sequential model.
 7. run godot_server_bridge.py to start the server that sends the video packages by UDP to a port that the godot script is listening to.
 8. open fake-it-till-you-make-it folder in Godot and run the game

---
**Running the FER + SER + Godot Bridge (`FER_SER_UI_godot_bridge.py`)**

The bridge requires the trained SER model weights, which are too large for git. Download `model.safetensors` from https://drive.switch.ch/index.php/s/5IOTren9xu2a6A2 and place it in the following folder (the folder already exists after cloning):

```
Multimodal-Emotion-Recognition/
└── MMUI/
    └── results/
        └── checkpoint-728/
            └── model.safetensors   ← place the file here
```

Then run:
```
python FER_SER_UI_godot_bridge.py
```

The script loads the SER model (Wav2Vec2) from `MMUI/results/checkpoint-728/` and the FER model (SqueezeNet) from `models/saved_weights/`. If the SER model file is missing it falls back to a dummy SER output — the app will still launch but speech emotion will not work correctly.

