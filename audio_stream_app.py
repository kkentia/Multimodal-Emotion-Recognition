
from dataclasses import dataclass
from queue import Queue, Empty
from collections import deque
from typing import Optional, Union
import time
import json
import numpy as np
import sounddevice as sd

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

    feature_type: str = "log_mel"
    n_mels: int = 64
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    normalize: bool = True

    emit_raw_waveform: bool = True
    emit_log_mel: bool = True
    emit_metadata: bool = True

    model_input_shape: Optional[tuple] = None
    output_format: str = "dict"


class AudioPackager:
    def __init__(self, config: AudioInterfaceConfig):
        self.cfg = config
        self.queue = Queue()
        self.buffer = deque()
        self.current_block = np.zeros((0, self.cfg.channels), dtype=self.cfg.dtype)
        self.max_samples = int(self.cfg.window_seconds * self.cfg.sample_rate)
        self.hop_samples = max(1, int(self.max_samples * (1 - self.cfg.overlap_ratio)))
        self.mel_transform = None
        if self.cfg.feature_type == "log_mel" and torchaudio is not None:
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.cfg.sample_rate,
                n_fft=self.cfg.n_fft,
                hop_length=self.cfg.hop_length,
                win_length=self.cfg.win_length,
                n_mels=self.cfg.n_mels,
            )

    def _resample_if_needed(self, audio: np.ndarray, sr_in: int) -> np.ndarray:
        if sr_in == self.cfg.sample_rate:
            return audio
        if torchaudio is None or torch is None:
            raise RuntimeError("torchaudio and torch are required for resampling")
        x = torch.tensor(audio.T, dtype=torch.float32)
        y = torchaudio.functional.resample(x, sr_in, self.cfg.sample_rate)
        return y.T.numpy()

    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        if not self.cfg.normalize:
            return audio
        peak = np.max(np.abs(audio)) + 1e-8
        return audio / peak

    def _vad_keep(self, audio: np.ndarray) -> bool:
        if not self.cfg.use_vad:
            return True
        energy = float(np.mean(audio ** 2))
        return energy >= self.cfg.vad_threshold

    def _log_mel(self, audio: np.ndarray):
        if self.mel_transform is None:
            raise RuntimeError("log-mel requires torchaudio")
        x = torch.tensor(audio.T, dtype=torch.float32)
        mel = self.mel_transform(x)
        return torch.log(mel + 1e-6).numpy()

    def package_window(self, audio: np.ndarray, start_time: float, end_time: float) -> dict:
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
        if self.cfg.emit_log_mel and self.cfg.feature_type == "log_mel":
            payload["log_mel"] = self._log_mel(audio)
        if self.cfg.model_input_shape is not None:
            payload["expected_input_shape"] = self.cfg.model_input_shape
        if not payload["vad_keep"]:
            payload["quality_flag"] = "silence_like"
        return payload

    def format_output(self, payload: dict):
        if self.cfg.output_format == "json":
            serializable = {}
            for k, v in payload.items():
                if isinstance(v, np.ndarray):
                    serializable[k] = v.tolist()
                else:
                    serializable[k] = v
            return json.dumps(serializable)
        return payload

    def push_audio_block(self, block: np.ndarray):
        if block.ndim == 1:
            block = block[:, None]
        self.current_block = np.vstack([self.current_block, block])

    def pop_windows(self):
        outputs = []
        while len(self.current_block) >= self.max_samples:
            window = self.current_block[:self.max_samples]
            self.current_block = self.current_block[self.hop_samples:]
            now = time.time()
            payload = self.package_window(window, now - self.cfg.window_seconds, now)
            outputs.append(self.format_output(payload))
        return outputs


class AudioStreamApp:
    def __init__(self, config: AudioInterfaceConfig):
        self.cfg = config
        self.packager = AudioPackager(config)
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
            print('Audio stream running. Press Ctrl+C to stop.')
            while self.running:
                try:
                    item = self.packager.queue.get(timeout=0.5)
                    yield item
                except Empty:
                    continue


if __name__ == "__main__":
    cfg = AudioInterfaceConfig()
    app = AudioStreamApp(cfg)
    for packet in app.run():
        print(packet if isinstance(packet, str) else {k: (v.shape if isinstance(v, np.ndarray) else v) for k, v in packet.items()})
