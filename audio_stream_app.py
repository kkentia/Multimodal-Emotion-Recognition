from dataclasses import dataclass
from queue import Queue, Empty
from pathlib import Path
from collections import deque
from typing import Optional, Union
import time
import json
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write

try:
    import torch
    import torchaudio
except Exception:
    torch = None
    torchaudio = None


@dataclass
class AudioInterfaceConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "float32"
    device: Optional[Union[int, str]] = None

    window_seconds: float = 3.0
    overlap_ratio: float = 0.5

    use_vad: bool = True
    vad_threshold: float = 0.01

    feature_type: str = "log_mel"   # "raw" or "log_mel"
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    normalize: bool = True

    emit_raw_waveform: bool = True
    emit_log_mel: bool = True
    emit_metadata: bool = True

    save_debug_wav: bool = True
    model_input_shape: Optional[tuple] = None
    output_format: str = "dict"     # "dict" or "json"


class AudioPackager:
    def __init__(self, config: AudioInterfaceConfig, model_path: Optional[str] = None):
        
        self.model = None
        self.feature_extractor = None
        if model_path:
            self.load_model(model_path)

    def load_model(self, model_path: str):
        """Load fine-tuned Wav2Vec2 emotion model"""
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.device = device
        print(f"Model loaded on {device}")

    def package_window(self, audio: np.ndarray, start_time: float, end_time: float) -> dict:
        raw_audio = audio.copy()
        audio = self._normalize(audio)

        payload = {
            "start_time": start_time,
            "end_time": end_time,
            "sample_rate": self.cfg.sample_rate,
            "channels": self.cfg.channels,
            "window_seconds": self.cfg.window_seconds,
            "window_samples": int(audio.shape[0]),
            "vad_keep": self._vad_keep(audio),
            "quality_flag": "ok",
        }

        if self.cfg.emit_raw_waveform:
            payload["waveform"] = audio

        if self.cfg.save_debug_wav:
            timestamp_ns = time.time_ns()
            filename = f"audio_window_{timestamp_ns}.wav"
            saved_path = self.save_window_to_wav(raw_audio, filename)
            payload["saved_wav"] = str(saved_path)

        # Model inference: only run if model is loaded and VAD passes
        if self.model is not None and payload["vad_keep"]:
            emotion_probs = self.run_emotion_inference(audio)
            payload["emotion_probs"] = emotion_probs
            payload["emotion_pred"] = emotion_probs.argmax().item()
            payload["emotion_conf"] = float(emotion_probs.max())

        if self.cfg.emit_log_mel and self.cfg.feature_type == "log_mel":
            payload["log_mel"] = self._log_mel(audio)

        if not payload["vad_keep"]:
            payload["quality_flag"] = "silence_like"

        return payload

    def run_emotion_inference(self, audio: np.ndarray) -> torch.Tensor:
        """Run Wav2Vec2 model inference on audio window"""
        # Convert to torch tensor and ensure correct shape
        if audio.ndim == 2:
            audio_mono = audio[:, 0]
        else:
            audio_mono = audio
        
        # Feature extraction matching training
        inputs = self.feature_extractor(
            audio_mono, 
            sampling_rate=self.cfg.sample_rate,
            return_tensors="pt",
            padding=True
        )
        
        # Move to model device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        return probs.cpu()


class AudioStreamApp:
    def __init__(self, config: AudioInterfaceConfig, model_path: Optional[str] = None):
        self.cfg = config
        self.packager = AudioPackager(config, model_path)  # Pass model path
        self.running = False

    def callback(self, indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        self.packager.push_audio_block(indata.copy())
        for payload in self.packager.pop_windows():
            self.packager.queue.put(payload)

    def run(self):
        self.running = True
        with sd.InputStream(
            device=self.cfg.device,
            channels=self.cfg.channels,
            samplerate=self.cfg.sample_rate,
            dtype=self.cfg.dtype,
            callback=self.callback,
        ):
            print("Audio stream running. Press Ctrl+C to stop.")
            while self.running:
                try:
                    item = self.packager.queue.get(timeout=0.5)
                    yield item
                except Empty:
                    continue


if __name__ == "__main__":
    cfg = AudioInterfaceConfig()
    app = AudioStreamApp(cfg)

    try:
        for packet in app.run():
            if isinstance(packet, str):
                print(packet)
            else:
                summary = {
                    k: (v.shape if isinstance(v, np.ndarray) else v)
                    for k, v in packet.items()
                }
                print(summary)
    except KeyboardInterrupt:
        print("Stopped.")
