import cv2
import socket
import json
import time
import math
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
STABLE_REQUIRED_FRAMES = 10  
CONF_THRESHOLD = 0.60 

# Must be 6 emotions matching the training script exactly!
CLASS_NAMES =["angry", "disgust", "fear", "happy", "neutral", "sad"]
MODEL_PATH = "models/saved_weights/best_static_mesh.pth"
TASK_PATH = "models/face_landmarker.task"

SPELLS = { 
    ("happy", "angry"): "Confusion",
    ("fear", "fear"): "Terror Strike",
    ("neutral", "neutral"): "Ice Shield",
    ("happy", "happy"): "Healing Aura",
    ("angry", "angry"): "Fireball",
    ("sad", "sad"): "Life Drain"
}

# ================== CLASSES ==================
class StaticMLP(nn.Module): #needs to match the model we trained so the weights know where and how to be used
    def __init__(self, input_size, num_classes):
        super(StaticMLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def forward(self, x): return self.network(x)

def calc_distance(p1, p2, w, h):
    return math.sqrt(((p1.x - p2.x)*w)**2 + ((p1.y - p2.y)*h)**2 + ((p1.z - p2.z)*w)**2)

def extract_features(landmarks, w, h):
    iod = calc_distance(landmarks[133], landmarks[362], w, h) + 1e-6
    return [
        calc_distance(landmarks[78], landmarks[308], w, h) / iod,
        calc_distance(landmarks[13], landmarks[14], w, h) / iod,
        calc_distance(landmarks[33], landmarks[133], w, h) / iod,
        calc_distance(landmarks[159], landmarks[145], w, h) / iod,
        calc_distance(landmarks[362], landmarks[263], w, h) / iod,
        calc_distance(landmarks[386], landmarks[374], w, h) / iod,
        calc_distance(landmarks[105], landmarks[159], w, h) / iod,
        calc_distance(landmarks[107], landmarks[33], w, h) / iod,
        calc_distance(landmarks[334], landmarks[386], w, h) / iod,
        calc_distance(landmarks[336], landmarks[263], w, h) / iod
    ]

# ================== UTILITIES ==================
def emotion_color(emotion):
    colors = {
        "happy": (80, 220, 120), "sad": (255, 140, 90),
        "angry": (80, 80, 255),  "fear": (180, 120, 255),
        "disgust": (180, 200, 80), "neutral": (200, 200, 210),
        "unknown": (180, 180, 180),
    }
    return colors.get(emotion.lower(), (255, 255, 255))

def fake_ser(): return ("angry", 0.78)

# ================== MAIN APP  ==================
def main():
    print("[INFO] Starting Godot Static Face Mesh Server...")
    
    device = torch.device("cpu")
    model = StaticMLP(10, len(CLASS_NAMES)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True)) #load weights
    model.eval() # no learning, just eval mode

    base_options = python.BaseOptions(model_asset_path=TASK_PATH)
    options = vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    stable_frames, last_combo, last_sent_time = 0, None, 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        results = detector.detect(mp_image)

        face_label, face_conf = "unknown", 0.0

        if len(results.face_landmarks) > 0:
            landmarks = results.face_landmarks[0]
            
            # Prediction
            feats = extract_features(landmarks, w, h)
            input_tensor = torch.tensor([feats], dtype=torch.float32).to(device)
            with torch.no_grad():
                probs = torch.softmax(model(input_tensor), dim=1)
                conf, pred_idx = torch.max(probs, dim=1)
            
            face_label = CLASS_NAMES[pred_idx.item()]
            face_conf = conf.item()

            #draw Box
            x_min = int(min([lm.x for lm in landmarks]) * w)
            y_min = int(min([lm.y for lm in landmarks]) * h)
            x_max = int(max([lm.x for lm in landmarks]) * w)
            y_max = int(max([lm.y for lm in landmarks]) * h)
            color = emotion_color(face_label)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
            cv2.putText(frame, f"{face_label.upper()} {face_conf:.2f}", (x_min, max(25, y_min - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

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