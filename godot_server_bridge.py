#godot_server_bridge.py

# common imports:
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import math
from collections import deque
import cv2
import urllib.request
import socket
import json
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision



# --------------------------------------------------CONV LSTM-------------------------------------------------------

'''
import cv2
import socket
import json
import time
import math
from collections import deque
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== CONFIG ==================
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242
VIDEO_PORT = 4243
SEND_INTERVAL = 1.0          
STABLE_REQUIRED_FRAMES = 5  # Lowered because 1 LSTM prediction inherently covers 10 frames
CONF_THRESHOLD = 0.60 

CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad"]
MODEL_PATH = "models/saved_weights/best_convlstm_model.pth"
TASK_PATH = "models/face_landmarker.task"

SPELLS = { 
    ("happy", "angry"): "Confusion", ("fear", "fear"): "Terror Strike",
    ("neutral", "neutral"): "Ice Shield", ("happy", "happy"): "Healing Aura",
    ("angry", "angry"): "Fireball", ("sad", "sad"): "Life Drain"
}

# ================== AI CLASSES ==================
class ConvLSTM1D(nn.Module):
    def __init__(self, input_features=10, hidden_size=64, num_classes=5):
        super(ConvLSTM1D, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=input_features, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden_size, num_layers=2, batch_first=True, dropout=0.3)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.relu(self.bn1(self.conv1d(x)))
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        return self.fc2(self.dropout(self.relu(self.fc1(lstm_out[:, -1, :]))))

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

def emotion_color(emo):
    colors = {"happy": (80,220,120), "sad": (255,140,90), "angry": (80,80,255), "fear": (180,120,255), "neutral": (200,200,210)}
    return colors.get(emo.lower(), (255, 255, 255))

def fake_ser(): return ("angry", 0.78)

# ================== MAIN LOOP ==================
def main():
    print("[INFO] Starting Temporal ConvLSTM Server...")
    
    device = torch.device("cpu")
    model = ConvLSTM1D().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    detector = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1))

    cap = cv2.VideoCapture(0)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    window = deque(maxlen=10) # Time series buffer!
    frame_count, stable_frames, last_sent_time = 0, 0, 0.0
    last_combo, face_label, face_conf = None, "unknown", 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        if len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]
            
            # Process every 2nd frame to match training speed
            if frame_count % 2 == 0:
                feats = extract_features(landmarks, w, h)
                window.append(feats)
                
                # Predict ONLY when we have 10 frames of history
                if len(window) == 10:
                    input_tensor = torch.tensor([list(window)], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        probs = torch.softmax(model(input_tensor), dim=1)
                        conf, pred_idx = torch.max(probs, dim=1)
                    
                    face_label = CLASS_NAMES[pred_idx.item()]
                    face_conf = conf.item()

            # Draw Box
            x_min = int(min([lm.x for lm in landmarks]) * w)
            y_min = int(min([lm.y for lm in landmarks]) * h)
            x_max = int(max([lm.x for lm in landmarks]) * w)
            y_max = int(max([lm.y for lm in landmarks]) * h)
            color = emotion_color(face_label)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{face_label.upper()} {face_conf:.2f}", (x_min, max(25, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        else:
            window.clear() # Reset sequence if face is lost!

        # Video Stream
        small_frame = cv2.resize(frame, (480, 360))
        _, jpg_buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        udp_sock.sendto(jpg_buffer.tobytes(), (GODOT_HOST, VIDEO_PORT))

        # Spell Logic
        speech_label, speech_conf = fake_ser()
        fused_conf = (face_conf + speech_conf) / 2.0
        spell = SPELLS.get((face_label.lower(), speech_label.lower()), None) if face_label != "unknown" else None

        combo = (face_label, speech_label)
        if combo == last_combo: stable_frames += 1
        else: stable_frames, last_combo = 0, combo

        now = time.time()
        if spell and fused_conf >= CONF_THRESHOLD and stable_frames >= STABLE_REQUIRED_FRAMES and (now - last_sent_time >= SEND_INTERVAL):
            payload = {
                "face_emotion": face_label.title(), "face_confidence": round(float(face_conf), 3),
                "speech_emotion": speech_label.title(), "speech_confidence": round(float(speech_conf), 3),
                "fused_confidence": round(float(fused_conf), 3), "spell": spell
            }
            udp_sock.sendto(json.dumps(payload).encode("utf-8"), (GODOT_HOST, GODOT_PORT))
            last_sent_time = now

if __name__ == "__main__":
    main()





'''

# ---------------------------------------------------- SQUEEZE NET -----------------------------------------------------------



# ==========================================
# 1. CONFIGURATION FOR GODOT
# ==========================================
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242
VIDEO_PORT = 4243

SEND_INTERVAL = 1.0          
STABLE_REQUIRED_FRAMES = 5  
CONF_THRESHOLD = 0.60 

# Friend's Model Paths (Windows format)
REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "models", "saved_weights", "best_squeezenet_mesh_full_3.pth")
TASK_PATH = os.path.join(REPO_ROOT, "models", "face_landmarker.task")

# Friend's 4 classes (Neutral is removed)
CLASSES = ["angry", "fear", "happy", "sad"]
NUM_CLASSES = len(CLASSES)

# Updated Spells for 4 emotions
SPELLS = { 
    ("happy", "angry"): "Confusion",
    ("fear", "fear"): "Terror Strike",
    ("happy", "happy"): "Healing Aura",
    ("angry", "angry"): "Fireball",
    ("sad", "sad"): "Life Drain"
}

# ==========================================
# 2. SETUP SQUEEZENET (Friend's Model)
# ==========================================
print("[INFO] Loading PyTorch SqueezeNet...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.squeezenet1_1()
model.classifier[1] = nn.Conv2d(512, NUM_CLASSES, kernel_size=(1, 1), stride=(1, 1))
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. SETUP MEDIAPIPE
# ==========================================
print("[INFO] Initializing MediaPipe...")
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)
TESSELATION = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

# ==========================================
# 4. UTILITIES
# ==========================================
def emotion_color(emotion):
    colors = {"happy": (80, 220, 120), "sad": (255, 140, 90), "angry": (80, 80, 255), "fear": (180, 120, 255), "unknown": (180, 180, 180)}
    return colors.get(emotion.lower(), (255, 255, 255))

def fake_ser(): 
    return ("angry", 0.78)

# ==========================================
# 5. MAIN GODOT SERVER LOOP
# ==========================================
def main():
    print("[INFO] Starting Visual Mesh Godot Server...")
    cap = cv2.VideoCapture(0)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    stable_frames = 0
    last_combo = None
    last_sent_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = detector.detect(mp_image)

        face_label = "unknown"
        face_conf = 0.0

        if mp_results.face_landmarks and len(mp_results.face_landmarks) > 0:
            face_landmarks = mp_results.face_landmarks[0]
            
            # --- FRIEND's VISUAL MESH DRAWING LOGIC ---
            black_bg = np.zeros((h, w, 3), dtype=np.uint8)
            points =[]
            
            for lm in face_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))
                
            for connection in TESSELATION:
                start_idx, end_idx = connection.start, connection.end
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(black_bg, points[start_idx], points[end_idx], (255, 255, 255), 1)
                    
            # --- PYTORCH INFERENCE ON THE DRAWN MESH ---
            ai_image = cv2.resize(black_bg, (224, 224))
            pil_image = Image.fromarray(ai_image)
            input_tensor = transform(pil_image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                probs = torch.nn.functional.softmax(output[0], dim=0)
                conf, pred_idx = torch.max(probs, dim=0)
                
            face_label = CLASSES[pred_idx.item()]
            face_conf = conf.item()

            # --- DRAW UI FOR GODOT ---
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, y1 = max(0, min(x_coords)-10), max(0, min(y_coords)-10)
            x2, y2 = min(w, max(x_coords)+10), min(h, max(y_coords)+10)
            
            color = emotion_color(face_label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{face_label.upper()} {face_conf:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Stream Video to Godot
        small_frame = cv2.resize(frame, (480, 360))
        _, jpg_buffer = cv2.imencode('.jpg', small_frame,[cv2.IMWRITE_JPEG_QUALITY, 70])
        udp_sock.sendto(jpg_buffer.tobytes(), (GODOT_HOST, VIDEO_PORT))

        # Spell Logic & Send Data
        speech_label, speech_conf = fake_ser()
        fused_conf = (face_conf + speech_conf) / 2.0
        
        spell = SPELLS.get((face_label.lower(), speech_label.lower()), None) if face_label != "unknown" else None
        combo = (face_label, speech_label)
        
        if combo == last_combo: stable_frames += 1
        else: stable_frames, last_combo = 0, combo

        now = time.time()
        if spell and fused_conf >= CONF_THRESHOLD and stable_frames >= STABLE_REQUIRED_FRAMES and (now - last_sent_time >= SEND_INTERVAL):
            payload = {
                "face_emotion": face_label.title(), "face_confidence": round(float(face_conf), 3),
                "speech_emotion": speech_label.title(), "speech_confidence": round(float(speech_conf), 3),
                "fused_confidence": round(float(fused_conf), 3), "spell": spell
            }
            udp_sock.sendto(json.dumps(payload).encode("utf-8"), (GODOT_HOST, GODOT_PORT))
            last_sent_time = now
            print(f"[INFO] Cast Spell: {spell}")

if __name__ == "__main__":
    main()