import cv2
import numpy as np
import socket
import json
import time
from collections import deque
from enum import Enum

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

#configure for godot
GODOT_HOST = "127.0.0.1"
GODOT_PORT = 4242

SEND_INTERVAL = 1.0          # minimum sec between sends
STABLE_REQUIRED_FRAMES = 15  # how long it must stay the same before sending frames
CONF_THRESHOLD = 0.60 

WINDOW_NAME = "Emotion Spell Interface"

# Path to the Resnet18 model
MODEL_PATH = "resnet18_emotion.pth"

#we need to check if these match with the emotions from the model 
CLASS_NAMES = ["angry", "fear", "happy", "neutral", "sad", "surprise"]

USE_GRAYSCALE_MODEL = False

#three modes for the prediction
class Mode(Enum):
    FER_ONLY = 1
    SER_ONLY = 2
    FUSED = 3

#example for spells we could include
SPELLS = {
    ("happy", "angry"): "fire_orb",
    ("sad", "fear"): "ice_wall",
    ("surprise", "happy"): "light_burst",
    ("neutral", "angry"): "shadow_push",
    ("happy", "happy"): "healing_wave",
    ("angry", "angry"): "thunder_strike",
    ("sad", "sad"): "mist_shield",
    ("neutral", "neutral"): "idle_aura",
}

#resnet model
class ResNetFER:
    def __init__(self, model_path, class_names, use_grayscale=False, device=None):
        self.class_names = class_names
        self.use_grayscale = use_grayscale
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = models.resnet18(weights=None)

        # If model was trained on single-channel images we can uncomment this behavior
        if self.use_grayscale:
            self.model.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, len(class_names))

        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        
        #this section must match how the model was trained
        if self.use_grayscale:
            self.transform = transforms.Compose([
                transforms.Grayscale(num_output_channels=1),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60)
        )

        if len(faces) == 0:
            return None

        # pick largest face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        x, y, w, h = faces[0]

        # add small padding
        pad = 10
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)

        face_crop = frame[y1:y2, x1:x2]
        return face_crop, (x1, y1, x2, y2)

    def predict(self, frame):
        detected = self.detect_face(frame)
        if detected is None:
            return ("unknown", 0.0, None)

        face_crop, bbox = detected

        # Convert OpenCV BGR -> RGB for PIL
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)

        input_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            conf, pred_idx = torch.max(probs, dim=1)

        label = self.class_names[pred_idx.item()]
        confidence = conf.item()

        return (label, confidence, bbox)

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
    
def fake_ser():
    # TODO: replace with real SER output
    return ("angry", 0.78)


def late_fusion(fer, ser):
    """
    Returns a fused confidence. Labels remain separate because
    the game spell is based on the pair (face, speech).
    """
    fused_conf = (fer[1] + ser[1]) / 2.0
    return fused_conf


def get_spell(face_label, speech_label):
    if face_label.lower() == "unknown":
        return None
    return SPELLS.get((face_label.lower(), speech_label.lower()), None)


def build_payload(face, fer_conf, speech, ser_conf, fused_conf, spell):
    return {
        "face_emotion": face,
        "face_confidence": round(float(fer_conf), 3),
        "speech_emotion": speech,
        "speech_confidence": round(float(ser_conf), 3),
        "fused_confidence": round(float(fused_conf), 3),
        "spell_key": f"{face}_{speech}",
        "spell": spell,
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
    try:
        fer_model = ResNetFER(
            model_path=MODEL_PATH,
            class_names=CLASS_NAMES,
            use_grayscale=USE_GRAYSCALE_MODEL
        )
        print(f"[INFO] FER model loaded from: {MODEL_PATH}")
        print(f"[INFO] Running on device: {fer_model.device}")
    except Exception as e:
        print(f"[ERROR] Failed to load FER model: {e}")
        return

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
            cached_fer = fer_model.predict(frame)

        face_label, face_conf, face_bbox = cached_fer
        fer = (face_label, face_conf)

        ser = fake_ser()
        speech_label, speech_conf = ser

        fused_conf = late_fusion(fer, ser)

        # 2) Build spell combo
        combo = (face_label, speech_label)
        spell = get_spell(face_label, speech_label)

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
                fused_conf, spell
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

        # 5) Build UI
        h, w = frame.shape[:2]
        panel_w = 420
        canvas_h = h

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
        draw_history(canvas, px1, 610, history)

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
    cv2.destroyAllWindows()
    print("[INFO] Clean exit.")


if __name__ == "__main__":
    main()