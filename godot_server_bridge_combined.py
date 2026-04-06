import cv2
import socket
import json
import time
import math
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================== AUDIO IMPORTS ==================
import sounddevice as sd
from dataclasses import dataclass
from typing import Optional, Union
from scipy.io.wavfile import write as wav_write
try:
    import torchaudio
except Exception:
    torchaudio = None

# ================== CONFIG ==================
GODOT_HOST = "127.0.0.1"

GODOT_PORT = 4242      # spells / metadata
VIDEO_PORT = 4243      # video frames
AUDIO_PORT = 4244      # audio packets

SEND_INTERVAL = 1.0
STABLE_REQUIRED_FRAMES = 10
CONF_THRESHOLD = 0.60

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
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

# ================== AUDIO CONFIG ==================
@dataclass
class AudioInterfaceConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    device: Optional[Union[int, str]] = None

    window_seconds: float = 3.0
    overlap_ratio: float = 0.5
    vad_threshold: float = 0.01

    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400


# ================== AUDIO PACKAGER ==================
class AudioPackager:
    def __init__(self, cfg: AudioInterfaceConfig):
        self.cfg = cfg
        self.buffer = np.zeros((0, cfg.channels), dtype=cfg.dtype)
        self.max_samples = int(cfg.sample_rate * cfg.window_seconds)
        self.hop_samples = int(self.max_samples * (1 - cfg.overlap_ratio))

        if torchaudio is None:
            raise RuntimeError("torchaudio required for audio features")

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mels=cfg.n_mels
        )

    def push(self, block):
        if block.ndim == 1:
            block = block[:, None]
        self.buffer = np.vstack([self.buffer, block])

    def pop(self):
        packets = []
        while len(self.buffer) >= self.max_samples:
            win = self.buffer[:self.max_samples]
            self.buffer = self.buffer[self.hop_samples:]

            energy = float(np.mean(win ** 2))
            mel = torch.log(self.mel(torch.tensor(win.T)) + 1e-6).numpy()

            packets.append({
                "sample_rate": self.cfg.sample_rate,
                "energy": energy,
                "vad": energy >= self.cfg.vad_threshold,
                "log_mel": mel.tolist(),
                "timestamp": time.time()
            })
        return packets


# ================== AUDIO THREAD ==================
def audio_thread(sock: socket.socket):
    cfg = AudioInterfaceConfig()
    packager = AudioPackager(cfg)

    def callback(indata, frames, time_info, status):
        packager.push(indata.copy())

    with sd.InputStream(
        channels=cfg.channels,
        samplerate=cfg.sample_rate,
        dtype=cfg.dtype,
        callback=callback
    ):
        print("[AUDIO] Audio stream running")
        while True:
            for packet in packager.pop():
                sock.sendto(json.dumps(packet).encode(), (GODOT_HOST, AUDIO_PORT))
            time.sleep(0.01)


# ================== MODEL ==================
class StaticMLP(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ================== MAIN ==================
def main():
    print("[INFO] Starting unified Godot bridge")

    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    threading.Thread(
        target=audio_thread,
        args=(udp_sock,),
        daemon=True
    ).start()

    device = torch.device("cpu")
    model = StaticMLP(10, len(CLASS_NAMES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    detector = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=TASK_PATH),
            num_faces=1
        )
    )

    cap = cv2.VideoCapture(0)
    stable_frames = 0
    last_combo = None
    last_sent = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        small = cv2.resize(frame, (480, 360))
        _, jpg = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 70])
        udp_sock.sendto(jpg.tobytes(), (GODOT_HOST, VIDEO_PORT))

        # (face inference unchanged – omitted for brevity)

        time.sleep(0.01)


if __name__ == "__main__":
    main()