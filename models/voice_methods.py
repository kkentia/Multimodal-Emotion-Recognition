import os
from typing import Dict, Tuple

import librosa
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 16000
MAX_LEN = 3 * SAMPLE_RATE
MODEL_NAME = "facebook/wav2vec2-base"
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(REPO_ROOT, "data", "raw", "ravdess")
DEFAULT_RESULTS_DIR = os.path.join(REPO_ROOT, "results")


def extract_label(file_path: str, emotion_map: Dict[str, str] = EMOTION_MAP) -> str:
    filename = os.path.basename(file_path)
    parts = filename.split("-")
    emotion_code = parts[2]
    return emotion_map[emotion_code]


def load_audio(file_path: str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return audio


def fix_length(audio: np.ndarray, max_len: int = MAX_LEN) -> np.ndarray:
    if len(audio) > max_len:
        return audio[:max_len]
    return np.pad(audio, (0, max_len - len(audio)))


def normalize(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio if peak == 0 else audio / peak


def preprocess_audio_file(file_path: str) -> np.ndarray:
    audio = load_audio(file_path, sample_rate=SAMPLE_RATE)
    audio = fix_length(audio, max_len=MAX_LEN)
    audio = normalize(audio)
    return audio.astype(np.float32)


def collect_audio_files(data_path: str = DATA_PATH) -> list[str]:
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path does not exist: {data_path}")

    files = []
    for root, _, filenames in os.walk(data_path):
        for filename in filenames:
            if filename.endswith(".wav"):
                files.append(os.path.join(root, filename))

    if not files:
        raise FileNotFoundError(f"No .wav files found in dataset path: {data_path}")

    return sorted(files)


def encode_labels(files: list[str]) -> Tuple[list[str], np.ndarray, LabelEncoder, np.ndarray]:
    labels = [extract_label(file_path) for file_path in files]
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(labels)
    class_counts = np.bincount(encoded_labels)

    if np.any(class_counts < 2):
        rare_labels = [
            label_encoder.inverse_transform([class_index])[0]
            for class_index, count in enumerate(class_counts)
            if count < 2
        ]
        raise ValueError(
            "Each class needs at least 2 samples for a stratified split. "
            f"Classes with too few samples: {rare_labels}"
        )

    return labels, encoded_labels, label_encoder, class_counts


def get_label_mappings(label_encoder: LabelEncoder) -> Tuple[Dict[int, str], Dict[str, int]]:
    id2label = {idx: label for idx, label in enumerate(label_encoder.classes_)}
    label2id = {label: idx for idx, label in id2label.items()}
    return id2label, label2id


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_from_label_encoder(
    label_encoder: LabelEncoder,
    model_name: str = MODEL_NAME,
) -> Wav2Vec2ForSequenceClassification:
    id2label, label2id = get_label_mappings(label_encoder)
    return Wav2Vec2ForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(label_encoder.classes_),
        id2label=id2label,
        label2id=label2id,
    )


def find_latest_checkpoint(results_dir: str = DEFAULT_RESULTS_DIR) -> str:
    if not os.path.isdir(results_dir):
        raise FileNotFoundError(
            f"Results directory not found: {results_dir}. "
            "Train the voice model first or choose a checkpoint folder manually."
        )

    checkpoints = []
    for entry in os.listdir(results_dir):
        if not entry.startswith("checkpoint-"):
            continue
        full_path = os.path.join(results_dir, entry)
        if os.path.isdir(full_path):
            step = entry.split("-", maxsplit=1)[-1]
            if step.isdigit():
                checkpoints.append((int(step), full_path))

    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint folders were found in {results_dir}. "
            "Expected folders like checkpoint-123."
        )

    checkpoints.sort(key=lambda item: item[0])
    return checkpoints[-1][1]


def resolve_model_path(model_path: str | None = None) -> str:
    if model_path:
        if not os.path.isdir(model_path):
            raise FileNotFoundError(f"Model directory not found: {model_path}")
        return model_path
    return find_latest_checkpoint()


def load_voice_model(
    model_path: str | None = None,
) -> Tuple[Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor, torch.device, str]:
    resolved_path = resolve_model_path(model_path)
    device = get_device()
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(resolved_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(resolved_path)
    model.to(device)
    model.eval()

    return model, feature_extractor, device, resolved_path
