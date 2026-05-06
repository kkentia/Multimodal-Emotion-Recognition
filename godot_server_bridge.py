# last working godot_server_bridge

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

# for speech-to-text model
import queue
import threading
import sounddevice as sd
import vosk
# ---------------------------------------------------- SQUEEZE NET -----------------------------------------------------------

# ==========================================
# 1. CONFIGURATION FOR GODOT
# ==========================================
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242
VIDEO_PORT = 4243

SEND_INTERVAL = 0.2          # send emotion data 5x/sec
CONF_THRESHOLD = 0.50        # only send if face confidence is meaningful

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
WEIGHTS_PATH = os.path.join(REPO_ROOT, "models", "saved_weights", "best_squeezenet_mesh_full_3.pth")
TASK_PATH = os.path.join(REPO_ROOT, "models", "face_landmarker.task")

#  4 classes (Neutral is removed)
CLASSES = ["angry", "fear", "happy", "sad"]
NUM_CLASSES = len(CLASSES)

# Spoken keywords that trigger spell casting in Godot
# (When a keyword is heard, Godot checks if it matches the held spell)
SPELL_KEYWORDS = ["ignite", "baffle", "restore", "freeze", "strike", "drain"]

# ==========================================
# 2. SETUP SQUEEZENET 
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
    colors = {"happy": (80, 220, 120), "sad": (255, 140, 90), "angry": (80, 80, 255),
              "fear": (180, 120, 255), "unknown": (180, 180, 180)}
    return colors.get(emotion.lower(), (255, 255, 255))


def fake_ser():
    """Dummy speech emotion recognizer — replace later with real model."""
    return ("angry", 0.78)


def fake_speech_keyword():
    """Returns the most recent detected keyword, then clears it."""
    global detected_keyword
    kw = detected_keyword
    detected_keyword = ""   # reset so it only fires once
    return kw


# ==========================================
# 5. MAIN GODOT SERVER LOOP
# ==========================================
def main():
    print("[INFO] Starting Visual Mesh Godot Server...")
    cap = cv2.VideoCapture(0)
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    last_sent_time = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mp_results = detector.detect(mp_image)

        face_label = "unknown"
        face_conf = 0.0

        if mp_results.face_landmarks and len(mp_results.face_landmarks) > 0:
            face_landmarks = mp_results.face_landmarks[0]

            # --- VISUAL MESH DRAWING ---
            black_bg = np.zeros((h, w, 3), dtype=np.uint8)
            points = []

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
            x1, y1 = max(0, min(x_coords) - 10), max(0, min(y_coords) - 10)
            x2, y2 = min(w, max(x_coords) + 10), min(h, max(y_coords) + 10)

            color = emotion_color(face_label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{face_label.upper()} {face_conf:.2f}", (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # === Stream Video to Godot ===
        small_frame = cv2.resize(frame, (480, 360))
        _, jpg_buffer = cv2.imencode('.jpg', small_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        udp_sock.sendto(jpg_buffer.tobytes(), (GODOT_HOST, VIDEO_PORT))

        # === ALWAYS send emotion data to Godot at fixed rate ===
        # (Godot decides which spell to hold/cast — we just stream raw data)
        now = time.time()
        if face_label != "unknown" and face_conf >= CONF_THRESHOLD and (now - last_sent_time >= SEND_INTERVAL):
            speech_label, speech_conf = fake_ser()
            spoken_keyword = fake_speech_keyword()

            payload = {
                "face_emotion": face_label.title(),
                "face_confidence": round(float(face_conf), 3),
                "speech_emotion": speech_label.title(),
                "speech_confidence": round(float(speech_conf), 3),
                "spell": spoken_keyword   # the keyword spoken (empty if none) — Godot uses to trigger cast
            }
            udp_sock.sendto(json.dumps(payload).encode("utf-8"), (GODOT_HOST, GODOT_PORT))
            last_sent_time = now
            print(f"[DATA] face={face_label} ({face_conf:.2f}) voice={speech_label} kw='{spoken_keyword}'")




# === SPEECH-TO-TEXT SETUP ================================================================

# ==========================================
# 6. SPEECH-TO-TEXT SETUP (clean version)
# ==========================================
import queue
import threading
import sounddevice as sd
import vosk
import audioop

print("[INFO] Available audio devices:")
print(sd.query_devices())
print(f"[INFO] Default input: {sd.default.device}")

VOSK_MODEL_PATH = os.path.join(REPO_ROOT, "vosk-model-small-en-us-0.15")

print("[INFO] Loading Vosk speech model...")
vosk_model = vosk.Model(VOSK_MODEL_PATH)
recognizer = vosk.KaldiRecognizer(vosk_model, 16000)

audio_queue = queue.Queue()
detected_keyword = ""
audio_chunks_received = 0


def audio_callback(indata, frames, time_info, status):
    global audio_chunks_received
    if status:
        print(f"[AUDIO STATUS] {status}")
    audio_chunks_received += 1
    if audio_chunks_received == 1:
        print("[AUDIO] ✅ First audio chunk received!")
    
    audio_bytes = bytes(indata)
    rms = audioop.rms(audio_bytes, 2)
    if audio_chunks_received % 10 == 0:
        print(f"[AUDIO] volume RMS: {rms}")
    
    audio_queue.put(audio_bytes)


def speech_thread():
    global detected_keyword
    print("[INFO] 🎤 Speech thread starting...")
    try:
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype="int16",
                               channels=1, callback=audio_callback):
            print("[INFO] 🎤 Microphone listening for spell keywords...")
            while True:
                data = audio_queue.get()
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    if text:
                        print(f"[VOSK FINAL] heard: '{text}'")
                        for kw in SPELL_KEYWORDS:
                            if kw in text:
                                detected_keyword = kw
                                print(f"🎯 Keyword detected: {kw}")
                                break
                else:
                    partial = json.loads(recognizer.PartialResult())
                    p_text = partial.get("partial", "")
                    if p_text:
                        print(f"[VOSK partial] {p_text}")
    except Exception as e:
        print(f"[ERROR] Speech thread crashed: {e}")
        import traceback
        traceback.print_exc()


def fake_speech_keyword():
    """Returns the most recent detected keyword, then clears it."""
    global detected_keyword
    kw = detected_keyword
    detected_keyword = ""
    return kw


# Start the speech thread
threading.Thread(target=speech_thread, daemon=True).start()


if __name__ == "__main__":
    main()