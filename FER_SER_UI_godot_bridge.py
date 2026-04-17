import cv2
import numpy as np
import socket
import json
import time
import threading
import os
from collections import deque
from enum import Enum

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

#configure for godot
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242

SEND_INTERVAL = 1.0          # minimum sec between sends
STABLE_REQUIRED_FRAMES = 15  # how long it must stay the same before sending frames
CONF_THRESHOLD = 0.0

WINDOW_NAME = "Emotion Spell Interface"

MODEL_PATH = "best_squeezenet_mesh_full_5_actors_split.pth"
TASK_PATH  = "models/face_landmarker.task"

CLASS_NAMES = ["angry", "fear", "happy", "sad"]

# Whisper config 
WHISPER_MODEL_SIZE   = "base"   # "tiny" | "base" | "small" | "medium" | "large"
WHISPER_INTERVAL     = 3.0      # seconds between transcription runs
WHISPER_LANGUAGE     = "en"     # set to None for auto-detect

# SER config 
SER_SAMPLE_RATE   = 16000
SER_AUDIO_SECONDS = 3  # rolling buffer length

# maps Wav2Vec2 labels → game emotion names
SER_LABEL_MAP = {
    "angry":     "angry",
    "fearful":   "fear",
    "happy":     "happy",
    "neutral":   "sad",
    "sad":       "sad",
    "surprised": "happy",
    "calm":      "sad",
    "disgust":   "angry",
}

def _find_ser_checkpoint():
    """Auto-detect the latest trained SER checkpoint."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    if not os.path.isdir(results_dir):
        return None
    checkpoints = [
        (int(e.split("-")[-1]), os.path.join(results_dir, e))
        for e in os.listdir(results_dir)
        if e.startswith("checkpoint-") and e.split("-")[-1].isdigit()
    ]
    return sorted(checkpoints)[-1][1] if checkpoints else None

# rolling audio buffer filled by background thread
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
        return SER_LABEL_MAP.get(raw_label, "neutral"), conf.item()
#

# Whisper transcription
_whisper_transcript = ""
_whisper_lock       = threading.Lock()

class WhisperTranscriber:
    """Runs Whisper in a background thread on the rolling audio buffer."""

    def __init__(self, model_size: str = WHISPER_MODEL_SIZE):
        print(f"[INFO] Loading faster-whisper model '{model_size}'...")
        device  = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"
        self._model   = _FasterWhisperModel(model_size, device=device, compute_type=compute)
        self._running = False
        self._thread  = None
        print("[INFO] Whisper model ready.")

    def _loop(self):
        global _whisper_transcript
        while self._running:
            with _audio_lock:
                audio = _audio_buffer.copy()

            # faster-whisper expects float32 at 16 kHz — matches our audio buffer exactly
            segments, _ = self._model.transcribe(
                audio,
                language=WHISPER_LANGUAGE,
                beam_size=5,
            )
            text = " ".join(seg.text for seg in segments).strip()

            with _whisper_lock:
                _whisper_transcript = text

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
#

#three modes for the prediction
class Mode(Enum):
    FER_ONLY = 1
    SER_ONLY = 2
    FUSED = 3

# Spells must match Godot's main_lvl.gd exactly:
# spoken_word AND fer_emotion AND ser_emotion -> spell name
SPELLS = {
    ("ignite",  "angry", "angry"): "Fireball",
    ("baffle",  "happy", "angry"): "Confusion",
    ("restore", "happy", "happy"): "Healing",
    ("freeze",  "sad",   "fear"):  "IceShard",
    ("drain",   "sad",   "sad"):   "ShadowDrain",
}

# The trigger word the player must say for each spell
SPELL_KEYWORDS = {"ignite", "baffle", "restore", "freeze", "strike", "drain"}

def extract_spoken_word(transcript: str) -> str:
    """Return the first spell keyword found in the transcript, or empty string."""
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

#UI configuration
def emotion_color(emotion):
    colors = {
        "happy": (80, 220, 120),
        "sad": (255, 140, 90),
        "angry": (80, 80, 255),
        "surprise": (80, 220, 255),
        "fear": (180, 120, 255),
        "neutral": (200, 200, 210),
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

    combo_text = f"{face_label.title()} + {speech_label.title()}"
    cv2.putText(img, f"Combo: {combo_text}", (x1 + 18, y1 + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 212, 220), 1, cv2.LINE_AA)

    spell_text = spell_name if spell_name else "No spell"
    cv2.putText(img, f"Spell: {spell_text}", (x1 + 18, y1 + 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (210, 212, 220), 1, cv2.LINE_AA)

    status = "READY" if ready else "WAITING"
    status_color = (80, 220, 120) if ready else (200, 200, 210)
    cv2.putText(img, f"Status: {status}", (x1 + 18, y1 + 132),
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
    bar_h = 90
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    img[:] = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)
    cv2.putText(img, text, (20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.95, color, 2)


def put_footer(img, text):
    h, w = img.shape[:2]
    cv2.putText(img, text, (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 2)

def draw_transcript_panel(img, x1, y1, x2, y2, transcript: str):
    """Renders the Whisper speech-to-text transcript in the side panel."""
    cv2.rectangle(img, (x1, y1), (x2, y2), (35, 36, 48), -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (70, 72, 90), 1)
    cv2.rectangle(img, (x1, y1), (x1 + 8, y2), (100, 200, 255), -1)

    cv2.putText(img, "Whisper Transcript", (x1 + 18, y1 + 28),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, (245, 245, 250), 1, cv2.LINE_AA)
    cv2.line(img, (x1 + 16, y1 + 40), (x2 - 16, y1 + 40), (70, 72, 90), 1)

    # Word-wrap the transcript to fit panel width (~28 chars per line)
    words      = transcript.split() if transcript else ["..."]
    lines, cur = [], ""
    for word in words:
        if len(cur) + len(word) + 1 > 28:
            lines.append(cur)
            cur = word
        else:
            cur = (cur + " " + word).strip()
    if cur:
        lines.append(cur)

    for i, line in enumerate(lines[:3]):   # max 3 lines
        cv2.putText(img, line, (x1 + 18, y1 + 65 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)
    
def fake_fer():
    return ("neutral", 0.75, None)

def fake_ser():
    return ("angry", 0.78)

def real_ser(ser_model: SERModel):
    with _audio_lock:
        audio = _audio_buffer.copy()
    return ser_model.predict(audio)


def late_fusion(fer, ser):
    """
    Returns a fused confidence. Labels remain separate because
    the game spell is based on the pair (face, speech).
    """
    fused_conf = (fer[1] + ser[1]) / 2.0
    return fused_conf


def get_spell(spoken_word, face_label, speech_label):
    if face_label.lower() == "unknown" or not spoken_word:
        return None
    return SPELLS.get((spoken_word.lower(), face_label.lower(), speech_label.lower()), None)


def build_payload(face, fer_conf, speech, ser_conf, fused_conf, spoken_word, spell):
    return {
        "face_emotion": face,
        "face_confidence": round(float(fer_conf), 3),
        "speech_emotion": speech,
        "speech_confidence": round(float(ser_conf), 3),
        "fused_confidence": round(float(fused_conf), 3),
        "spoken_word": spoken_word,
        "spell": spell if spell else "",
        "timestamp": time.time()
    }


def send_to_godot_udp(sock, payload):
    data = json.dumps(payload).encode("utf-8")
    sock.sendto(data, (GODOT_HOST, GODOT_PORT))


def open_camera():
    for idx in [0, 1, 2, 3]:
        print(f"[INFO] Trying camera index {idx}...")
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

    # Load FER model
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

    # Load SER model if checkpoint exists, else use fake_ser
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

    # Load Whisper transcriber (requires audio stream to be running)
    whisper_transcriber = None
    if WHISPER_AVAILABLE:
        if audio_stream is None:
            # Whisper still needs the audio stream even if SER is fake
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

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    mode = Mode.FUSED
    history = deque(maxlen=10)

    last_combo = None
    stable_frames = 0
    last_sent_time = 0.0

    # optional caching to reduce inference load
    frame_count = 0
    cached_fer = ("unknown", 0.0, None)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[ERROR] Failed to grab frame.")
            break

        frame = cv2.flip(frame, 1)

        # 1) Get model outputs
        frame_count += 1

        # Run FER every frame. Change to % 2 or % 3 if you want more speed.
        if frame_count % 1 == 0:
            cached_fer = fer_model.predict(frame) if fer_model else fake_fer()

        face_label, face_conf, face_bbox = cached_fer
        fer = (face_label, face_conf)

        ser = real_ser(ser_model) if ser_model else fake_ser()
        speech_label, speech_conf = ser

        # Get Whisper transcript and extract spoken keyword
        transcript   = get_whisper_transcript()
        spoken_word  = extract_spoken_word(transcript)

        fused_conf = late_fusion(fer, ser)

        # 2) Build spell combo (spoken word + face + speech must all match)
        combo = (spoken_word, face_label, speech_label)
        spell = get_spell(spoken_word, face_label, speech_label)

        # 3) Stability check
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

        # 4) Send event to Godot if stable and not too frequent
        now = time.time()
        if ready and (now - last_sent_time >= SEND_INTERVAL):
            payload = build_payload(
                face_label, face_conf,
                speech_label, speech_conf,
                fused_conf, spoken_word, spell
            )
            send_to_godot_udp(udp_sock, payload)
            last_sent_time = now
            history.appendleft(f"CAST: {spell}")
            print("[INFO] Sent to Godot:", payload)

        # Draw FER face box
        if face_bbox is not None:
            x1, y1, x2, y2 = face_bbox
            color = emotion_color(face_label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{face_label} {face_conf:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA
            )

        TARGET_W = 800
        h_orig, w_orig = frame.shape[:2]
        scale = TARGET_W / w_orig
        frame = cv2.resize(frame, (TARGET_W, int(h_orig * scale)))

        h, w = frame.shape[:2]
        panel_w = 320
        canvas_h = max(h, 720)

        canvas = np.zeros((canvas_h, w + panel_w, 3), dtype=np.uint8)
        canvas[:, :] = (30, 30, 30)
        canvas[:h, :w] = frame

        left_view = canvas[:h, :w]

        if mode == Mode.FER_ONLY:
            hud_text = f"FER: {face_label.upper()} ({face_conf:.2f})"
            hud_color = emotion_color(face_label)
        elif mode == Mode.SER_ONLY:
            hud_text = f"SER: {speech_label.upper()} ({speech_conf:.2f})"
            hud_color = emotion_color(speech_label)
        else:
            spell_name = spell if spell else "none"
            hud_text = f"FUSED: {face_label.upper()} + {speech_label.upper()} -> {spell_name}"
            hud_color = (255, 170, 60)

        put_hud(left_view, hud_text, hud_color)
        put_footer(left_view, "Keys: 1=FER  2=SER  3=FUSED  Q/Esc=quit")

        px1, px2 = w + 20, w + panel_w - 20

        draw_card(canvas, px1, 20, px2, 140, "FER", face_label, face_conf, emotion_color(face_label))
        draw_card(canvas, px1, 155, px2, 275, "SER", speech_label, speech_conf, emotion_color(speech_label))

        fused_label = f"{face_label}+{speech_label}"
        draw_card(canvas, px1, 290, px2, 410, "FUSED", fused_label, fused_conf, (255, 170, 60))

        draw_spell_panel(canvas, px1, 425, px2, 575, face_label, speech_label, spell, ready)
        draw_history(canvas, px1, 615, history)

        if whisper_transcriber is not None:
            draw_transcript_panel(canvas, px1, canvas_h - 140, px2, canvas_h - 10, transcript)

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            print("[INFO] Quit key pressed.")
            break
        elif key == ord("1"):
            mode = Mode.FER_ONLY
        elif key == ord("2"):
            mode = Mode.SER_ONLY
        elif key == ord("3"):
            mode = Mode.FUSED

    udp_sock.close()
    cap.release()
    if whisper_transcriber:
        whisper_transcriber.stop()
    if audio_stream:
        audio_stream.stop()
    cv2.destroyAllWindows()
    print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()