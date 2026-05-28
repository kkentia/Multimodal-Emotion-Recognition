import cv2
import numpy as np
import socket
import json
import time
import threading
import os
from collections import deque
from enum import Enum
import csv

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import sounddevice as sd
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from faster_whisper import WhisperModel as _FasterWhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("[WARN] faster-whisper not installed. Transcription disabled. Run: pip install faster-whisper")

GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242
VIDEO_PORT = 4243

# Set to a specific index to force a camera (e.g. 1 to skip the phone).
# Leave as None to auto-detect (uses first available camera).
CAMERA_INDEX = None

SEND_INTERVAL = 1.0
STABLE_REQUIRED_FRAMES = 15
CONF_THRESHOLD = 0.0

WINDOW_NAME = "Emotion Spell Interface"

MODEL_PATH = "models/saved_weights/best_squeezenet_mesh_full_5_actors_split.pth"
TASK_PATH  = "models/face_landmarker.task"

CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad"]

WHISPER_MODEL_SIZE = "base"
WHISPER_INTERVAL   = 3.0
WHISPER_LANGUAGE   = "en"

SER_SAMPLE_RATE   = 16000
SER_AUDIO_SECONDS = 3

SER_LABEL_MAP = {
    "angry":     "angry",
    "fearful":   "fear",
    "happy":     "happy",
    "neutral":   "neutral",
    "sad":       "sad",
    "surprised": "happy",
    "calm":      "neutral",
    "disgust":   "angry",
}

PERFORMANCE_LOG = "performance_log.csv"
LATENCY_WINDOW = 200


    # Initialize the CSV with a header row, once at startup
def init_performance_log():
    with open(PERFORMANCE_LOG, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
    "frame", "timestamp",
    "fer_ms", "ser_ms", "whisper_ms", "whisper_inference_ms",
    "fusion_ms", "udp_send_ms", "total_ms",
    "face_emotion", "face_conf",
    "speech_emotion", "speech_conf",
    "spoken_word", "spell", "ready"
])
        
    
def _find_ser_checkpoint():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "MMUI", "results")
    if not os.path.isdir(results_dir):
        return None
    checkpoints = [
        (int(e.split("-")[-1]), os.path.join(results_dir, e))
        for e in os.listdir(results_dir)
        if e.startswith("checkpoint-") and e.split("-")[-1].isdigit()
    ]
    return sorted(checkpoints)[-1][1] if checkpoints else None

_audio_buffer = np.zeros(SER_AUDIO_SECONDS * SER_SAMPLE_RATE, dtype=np.float32)
_audio_lock   = threading.Lock()

def _audio_callback(indata, frames, time_info, status):
    global _audio_buffer
    with _audio_lock:
        _audio_buffer = np.roll(_audio_buffer, -frames)
        _audio_buffer[-frames:] = indata[:, 0]

def start_audio_stream():
    stream = sd.InputStream(
        samplerate=SER_SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=_audio_callback,
    )
    stream.start()
    return stream

class SERModel:
    def __init__(self, checkpoint_path):
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_path)
        self.model     = Wav2Vec2ForSequenceClassification.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, audio: np.ndarray):
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        inputs = self.extractor(
            audio, sampling_rate=SER_SAMPLE_RATE,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.model(inputs.input_values.to(self.device)).logits
            probs  = torch.softmax(logits, dim=1)
            conf, idx = torch.max(probs, dim=1)
        raw_label = self.model.config.id2label[idx.item()]
        return SER_LABEL_MAP.get(raw_label, "sad"), conf.item()

_whisper_transcript = ""
_whisper_lock       = threading.Lock()
_whisper_last_latency_ms = 0.0
_whisper_call_count = 0

class WhisperTranscriber:
    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        print(f"[INFO] Loading faster-whisper model '{model_size}'...")
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._model   = _FasterWhisperModel(model_size, device=device, compute_type=compute)
        self._running = False
        self._thread  = None
        print("[INFO] Whisper model ready.")

    def _loop(self):
        global _whisper_transcript, _whisper_last_latency_ms
        while self._running:
            with _audio_lock:
                audio = _audio_buffer.copy()
            t0 = time.perf_counter()
            segments, _ = self._model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                beam_size=5,
            )
            text = " ".join(seg.text for seg in segments).strip()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            with _whisper_lock:
                _whisper_transcript = text
                _whisper_last_latency_ms = elapsed_ms
            time.sleep(WHISPER_INTERVAL)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

def get_whisper_transcript() -> str:
    with _whisper_lock:
        return _whisper_transcript
    
def get_whisper_latency() -> float:
    with _whisper_lock:
        return _whisper_last_latency_ms
    
class Mode(Enum):
    FER_ONLY = 1
    SER_ONLY = 2
    FUSED = 3

SPELLS = {
    ("ignite",  "angry", "angry"): "Fireball",
    ("baffle",  "happy", "angry"): "Confusion",
    ("restore", "happy", "happy"): "Healing",
    ("freeze",  "sad",   "fear"):  "Ice Shard",
    ("strike",  "fear",  "angry"): "Lightning",
    ("drain",   "sad",   "sad"):   "Shadow Drain",
}

SPELL_KEYWORDS = {"ignite", "baffle", "restore", "freeze", "strike", "drain"}

def extract_spoken_word(transcript: str) -> str:
    text_lower = transcript.lower()
    for word in SPELL_KEYWORDS:
        if word in text_lower:
            return word
    return ""

class SqueezeNetFER:
    def __init__(self, model_path, class_names, task_path, device=None):
        self.class_names = class_names
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.squeezenet1_1(weights=None)
        self.model.classifier[1] = nn.Conv2d(512, len(class_names), kernel_size=(1, 1), stride=(1, 1))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        base_options = python.BaseOptions(model_asset_path=task_path)
        self.detector = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(base_options=base_options, num_faces=1)
        )
        self.tesselation = vision.FaceLandmarksConnections.FACE_LANDMARKS_TESSELATION

    def predict(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.detector.detect(mp_image)

        if not results.face_landmarks:
            return ("unknown", 0.0, None)

        landmarks = results.face_landmarks[0]
        points = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

        black_bg = np.zeros((h, w, 3), dtype=np.uint8)
        for conn in self.tesselation:
            s, e = conn.start, conn.end
            if s < len(points) and e < len(points):
                cv2.line(black_bg, points[s], points[e], (255, 255, 255), 1)

        pil_image = Image.fromarray(cv2.resize(black_bg, (224, 224)))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = torch.softmax(self.model(input_tensor), dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        bbox = (max(0, min(x_coords) - 10), max(0, min(y_coords) - 10),
                min(w, max(x_coords) + 10), min(h, max(y_coords) + 10))

        return (self.class_names[pred_idx.item()], conf.item(), bbox)

def emotion_color(emotion):
    colors = {
        "happy":   (80, 220, 120),
        "sad":     (255, 140, 90),
        "angry":   (80, 80, 255),
        "fear":    (180, 120, 255),
        "unknown": (180, 180, 180),
    }
    return colors.get(emotion.lower(), (255, 255, 255))

def draw_conf_bar(img, x, y, w, h, value, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 62, 75), -1)
    fill_w = int(w * max(0.0, min(1.0, value)))
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (95, 98, 120), 1)

def draw_card(img, x1, y1, x2, y2, title, label, conf, accent):
    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 36, 48), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 72, 90), 1)
    cv2.rectangle(img, (x1, y1), (x1 + 8, y2), accent, -1)
    cv2.putText(img, title, (x1 + 18, y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 245, 250), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 42), (x2 - 16, y1 + 42), (70, 72, 90), 1)
    cv2.putText(img, f"Label: {label.title()}", (x1 + 18, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (210, 212, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"Confidence: {conf:.2f}", (x1 + 18, y1 + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (210, 212, 220), 1, cv2.LINE_AA)
    draw_conf_bar(img, x1 + 18, y1 + 115, (x2 - x1) - 36, 14, conf, accent)

def draw_spell_panel(img, x1, y1, x2, y2, face_label, speech_label, spell_name, ready):
    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 36, 48), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 72, 90), 1)
    cv2.rectangle(img, (x1, y1), (x1 + 8, y2), (255, 170, 60), -1)
    cv2.putText(img, "Spell Fusion", (x1 + 18, y1 + 30),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (245, 245, 250), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 42), (x2 - 16, y1 + 42), (70, 72, 90), 1)
    cv2.putText(img, f"Combo: {face_label.title()} + {speech_label.title()}", (x1 + 18, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 212, 220), 1, cv2.LINE_AA)
    cv2.putText(img, f"Spell: {spell_name if spell_name else 'No spell'}", (x1 + 18, y1 + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 212, 220), 1, cv2.LINE_AA)
    status_color = (80, 220, 120) if ready else (200, 200, 210)
    cv2.putText(img, f"Status: {'READY' if ready else 'WAITING'}", (x1 + 18, y1 + 132),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)

def draw_history(img, x, y, history):
    cv2.putText(img, "Recent Spells", (x, y),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (245, 245, 250), 1, cv2.LINE_AA)
    y += 20
    for i, item in enumerate(list(history)[:6]):
        yy = y + i * 34
        cv2.rectangle(img, (x, yy), (x + 260, yy + 24), (45, 48, 66), -1)
        cv2.rectangle(img, (x, yy), (x + 260, yy + 24), (80, 84, 104), 1)
        cv2.putText(img, item, (x + 10, yy + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (220, 220, 228), 1, cv2.LINE_AA)

def put_hud(img, text, color):
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.putText(img, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)

def put_footer(img, text):
    h, w = img.shape[:2]
    cv2.putText(img, text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

def draw_transcript_panel(img, x1, y1, x2, y2, transcript: str):
    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 36, 48), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 72, 90), 1)
    cv2.rectangle(img, (x1, y1), (x1 + 8, y2), (100, 200, 255), -1)
    cv2.putText(img, "Whisper Transcript", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (245, 245, 250), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 40), (x2 - 16, y1 + 40), (70, 72, 90), 1)
    words = transcript.split() if transcript else ["..."]
    lines, cur = [], ""
    for word in words:
        if len(cur) + len(word) + 1 > 28:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)
    for i, line in enumerate(lines[:3]):
        cv2.putText(img, line, (x1 + 18, y1 + 65 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

def fake_fer():
    return ("sad", 0.75, None)

def fake_ser():
    return ("angry", 0.78)

def real_ser(ser_model: SERModel):
    with _audio_lock:
        audio = _audio_buffer.copy()
    return ser_model.predict(audio)

def late_fusion(fer, ser):
    return (fer[1] + ser[1]) / 2.0

def get_spell(spoken_word, face_label, speech_label):
    if face_label.lower() == "unknown" or not spoken_word:
        return None
    return SPELLS.get((spoken_word.lower(), face_label.lower(), speech_label.lower()), None)

def build_payload(face, fer_conf, speech, ser_conf, fused_conf, spoken_word, spell, transcript):
    return {
        "face_emotion":       face,
        "face_confidence":    round(float(fer_conf), 3),
        "speech_emotion":     speech,
        "speech_confidence":  round(float(ser_conf), 3),
        "fused_confidence":   round(float(fused_conf), 3),
        "spoken_word":        spoken_word,
        "spell":              spell if spell else "",
        "transcript":         transcript,
        "timestamp":          time.time()
    }

def send_to_godot_udp(sock, payload):
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, (GODOT_HOST, GODOT_PORT))

def open_camera():
    print("[INFO] Available cameras:")
    for idx in range(4):
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            print(f"  [{idx}] {'(found)' if ret else '(no frame)'}")
        cap.release()

    indices = [CAMERA_INDEX] if CAMERA_INDEX is not None else [0, 1, 2, 3]
    for idx in indices:
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[INFO] Using camera index {idx}")
            return cap
        cap.release()
    return None

def main():
    print("[INFO] Starting interface...")
    
    
    init_performance_log()
    fer_history = deque(maxlen=LATENCY_WINDOW)
    ser_history = deque(maxlen=LATENCY_WINDOW)
    whisper_history = deque(maxlen=LATENCY_WINDOW)
    total_history = deque(maxlen=LATENCY_WINDOW)
    
    
    fer_model = None
    try:
        fer_model = SqueezeNetFER(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            task_path=TASK_PATH,
        )
        print(f"[INFO] SqueezeNet FER model loaded from: {MODEL_PATH}")
        print(f"[INFO] Running on device: {fer_model.device}")
    except Exception as e:
        print(f"[WARN] FER model not found, using fake FER: {e}")

    ser_model    = None
    audio_stream = None
    checkpoint   = _find_ser_checkpoint()
    if checkpoint:
        try:
            ser_model    = SERModel(checkpoint)
            audio_stream = start_audio_stream()
            print(f"[INFO] SER model loaded: {checkpoint}")
        except Exception as e:
            print(f"[WARN] SER load failed, using fake SER: {e}")
    else:
        print("[WARN] No SER checkpoint found, using fake SER")

    whisper_transcriber = None
    if WHISPER_AVAILABLE:
        if audio_stream is None:
            audio_stream = start_audio_stream()
        try:
            whisper_transcriber = WhisperTranscriber(WHISPER_MODEL_SIZE)
            whisper_transcriber.start()
            print("[INFO] Whisper transcription started.")
        except Exception as e:
            print(f"[WARN] Whisper failed to start: {e}")

    cap = open_camera()
    if cap is None:
        print("[ERROR] Could not open webcam.")
        return

    udp_sock   = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)
    mode = Mode.FUSED
    history = deque(maxlen=10)

    last_combo    = None
    stable_frames = 0
    last_sent_time = 0.0
    frame_count   = 0
    cached_fer    = ("unknown", 0.0, None)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)
        frame_count += 1
        total_start = time.perf_counter()

        # === 1. FER timing ===
        fer_start = time.perf_counter()
        if frame_count % 1 == 0:
            cached_fer = fer_model.predict(frame) if fer_model else fake_fer()
        fer_latency = (time.perf_counter() - fer_start) * 1000.0

        face_label, face_conf, face_bbox = cached_fer
        fer = (face_label, face_conf)

        # === 2. SER timing ===
        ser_start = time.perf_counter()
        ser = real_ser(ser_model) if ser_model else fake_ser()
        ser_latency = (time.perf_counter() - ser_start) * 1000.0
        speech_label, speech_conf = ser

        # === 3. Whisper read timing (just the lookup; transcription runs on its own thread) ===
        whisper_start = time.perf_counter()
        transcript = get_whisper_transcript()
        spoken_word = extract_spoken_word(transcript)
        whisper_latency = (time.perf_counter() - whisper_start) * 1000.0

        # === 4. Fusion timing ===
        fusion_start = time.perf_counter()
        fused_conf = late_fusion(fer, ser)
        combo = (spoken_word, face_label, speech_label)
        spell = get_spell(spoken_word, face_label, speech_label)
        if combo == last_combo:
            stable_frames += 1
        else:
            stable_frames = 0
            last_combo = combo
        ready = (
            spell is not None
            and fused_conf >= CONF_THRESHOLD
            and stable_frames >= STABLE_REQUIRED_FRAMES
        )
        fusion_latency = (time.perf_counter() - fusion_start) * 1000.0

        # === 5. UDP send timing ===
        udp_latency = 0.0
        now = time.time()
        if now - last_sent_time >= SEND_INTERVAL:
            udp_start = time.perf_counter()
            payload = build_payload(
                face_label, face_conf,
                speech_label, speech_conf,
                fused_conf, spoken_word, spell, transcript
            )
            send_to_godot_udp(udp_sock, payload)
            udp_latency = (time.perf_counter() - udp_start) * 1000.0
            last_sent_time = now
            if ready and spell:
                history.appendleft(f"CAST: {spell}")
            print("[INFO] Sent to Godot:", payload)

        total_latency = (time.perf_counter() - total_start) * 1000.0

        # === LOG THE ROW ===
        fer_history.append(fer_latency)
        ser_history.append(ser_latency)
        whisper_history.append(whisper_latency)
        total_history.append(total_latency)

        with open(PERFORMANCE_LOG, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                frame_count, time.time(),
                round(fer_latency, 3),
                round(ser_latency, 3),
                round(whisper_latency, 3),
                round(get_whisper_latency(), 3),   # ← whisper_inference_ms
                round(fusion_latency, 3),
                round(udp_latency, 3),
                round(total_latency, 3),
                face_label, round(face_conf, 3),
                speech_label, round(speech_conf, 3),
                spoken_word, spell or "", ready,
            ])

        # === Periodic live summary in the console ===
        if frame_count % 60 == 0 and fer_history:
            print(f"[STATS] frames={frame_count}  "
                f"FER mean={np.mean(fer_history):.1f}ms  "
                f"SER mean={np.mean(ser_history):.1f}ms  "
                f"Whisper read mean={np.mean(whisper_history):.1f}ms  "
                f"Total mean={np.mean(total_history):.1f}ms")

        # ---------------------------------------------------------
        
        if frame_count % 1 == 0:
            cached_fer = fer_model.predict(frame) if fer_model else fake_fer()

        face_label, face_conf, face_bbox = cached_fer
        fer = (face_label, face_conf)

        ser = real_ser(ser_model) if ser_model else fake_ser()
        speech_label, speech_conf = ser

        transcript  = get_whisper_transcript()
        spoken_word = extract_spoken_word(transcript)

        fused_conf = late_fusion(fer, ser)

        combo = (spoken_word, face_label, speech_label)
        spell = get_spell(spoken_word, face_label, speech_label)

        if combo == last_combo:
            stable_frames += 1
        else:
            stable_frames = 0
            last_combo = combo

        ready = (
            spell is not None
            and fused_conf >= CONF_THRESHOLD
            and stable_frames >= STABLE_REQUIRED_FRAMES
        )

        now = time.time()
        if now - last_sent_time >= SEND_INTERVAL:
            payload = build_payload(
                face_label, face_conf,
                speech_label, speech_conf,
                fused_conf, spoken_word, spell, transcript
            )
            send_to_godot_udp(udp_sock, payload)
            last_sent_time = now
            if ready and spell:
                history.appendleft(f"CAST: {spell}")
            print("[INFO] Sent to Godot:", payload)

        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            color = emotion_color(face_label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{face_label} {face_conf:.2f}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        _, jpg = cv2.imencode('.jpg', cv2.resize(frame, (320, 240)), [cv2.IMWRITE_JPEG_QUALITY, 50])
        video_sock.sendto(jpg.tobytes(), (GODOT_HOST, VIDEO_PORT))

        TARGET_W = 800
        h_orig, w_orig = frame.shape[:2]
        scale = TARGET_W / w_orig
        frame = cv2.resize(frame, (TARGET_W, int(h_orig * scale)))

        h, w = frame.shape[:2]
        panel_w  = 320
        canvas_h = max(h, 720)

        canvas = np.zeros((canvas_h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :] = (30, 30, 30)
        canvas[:h, :w] = frame

        left_view = canvas[:h, :w]

        if mode == Mode.FER_ONLY:
            hud_text  = f"FER: {face_label.upper()} ({face_conf:.2f})"
            hud_color = emotion_color(face_label)
        elif mode == Mode.SER_ONLY:
            hud_text  = f"SER: {speech_label.upper()} ({speech_conf:.2f})"
            hud_color = emotion_color(speech_label)
        else:
            spell_name = spell if spell else "none"
            hud_text  = f"FUSED: {face_label.upper()} + {speech_label.upper()} -> {spell_name}"
            hud_color = (255, 170, 60)

        put_hud(left_view, hud_text, hud_color)
        put_footer(left_view, "Keys: 1=FER  2=SER  3=FUSED  Q/Esc=quit")

        px1, px2 = w + 20, w + panel_w - 20

        draw_card(canvas, px1, 20,  px2, 140, "FER",   face_label,   face_conf,   emotion_color(face_label))
        draw_card(canvas, px1, 155, px2, 275, "SER",   speech_label, speech_conf, emotion_color(speech_label))
        draw_card(canvas, px1, 290, px2, 410, "FUSED", f"{face_label}+{speech_label}", fused_conf, (255, 170, 60))

        draw_spell_panel(canvas, px1, 425, px2, 575, face_label, speech_label, spell, ready)
        draw_history(canvas, px1, 615, history)

        if whisper_transcriber is not None:
            draw_transcript_panel(canvas, px1, canvas_h - 140, px2, canvas_h - 10, transcript)

        # cv2.imshow(WINDOW_NAME, canvas)

        # key = cv2.waitKey(1) & 0xFF
        # if key in (ord("q"), 27):
        #     print("[INFO] Quit key pressed.")
        #     break
        # elif key == ord("1"):
        #     mode = Mode.FER_ONLY
        # elif key == ord("2"):
        #     mode = Mode.SER_ONLY
        # elif key == ord("3"):
        #     mode = Mode.FUSED

    udp_sock.close()
    video_sock.close()
    cap.release()
    if whisper_transcriber:
        whisper_transcriber.stop()
    if audio_stream:
        audio_stream.stop()
    cv2.destroyAllWindows()
    print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()