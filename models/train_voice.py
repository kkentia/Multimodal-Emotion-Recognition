import os
from dataclasses import dataclass
import warnings

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchaudio
from IPython.display import Audio
from datasets import Dataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch import nn
from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

from voice_methods import (
    DATA_PATH,
    MAX_LEN,
    MODEL_NAME,
    SAMPLE_RATE,
    build_model_from_label_encoder,
    collect_audio_files,
    encode_labels,
    get_device,
    preprocess_audio_file,
)

warnings.filterwarnings("ignore")


@dataclass
class TrainConfig:
    output_dir: str = "./results"
    logging_dir: str = "./logs"
    learning_rate: float = 1e-5
    train_batch_size: int = 16
    eval_batch_size: int = 16
    num_train_epochs: int = 8
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_accumulation_steps: int = 1
    freeze_feature_encoder: bool = True


def prepare_input(batch, feature_extractor):
    processed_audio = [preprocess_audio_file(file_path) for file_path in batch["file_path"]]
    output = feature_extractor(
        processed_audio,
        sampling_rate=SAMPLE_RATE,
        padding=True,
        truncation=True,
        max_length=MAX_LEN,
    )
    output["label"] = batch["label"]
    return output


def compute_metrics(pred):
    preds = np.argmax(pred.predictions, axis=1)
    return {"accuracy": accuracy_score(pred.label_ids, preds)}


def build_datasets(files, labels_encoded, feature_extractor):
    X_train, X_test, y_train, y_test = train_test_split(
        files,
        labels_encoded,
        test_size=0.2,
        random_state=42,
        stratify=labels_encoded,
    )

    train_dataset = Dataset.from_dict({"file_path": X_train, "label": y_train})
    test_dataset = Dataset.from_dict({"file_path": X_test, "label": y_test})

    train_dataset = train_dataset.map(
        lambda batch: prepare_input(batch, feature_extractor),
        batched=True,
        remove_columns=["file_path"],
    )
    test_dataset = test_dataset.map(
        lambda batch: prepare_input(batch, feature_extractor),
        batched=True,
        remove_columns=["file_path"],
    )

    train_dataset.set_format(type="torch")
    test_dataset.set_format(type="torch")
    return train_dataset, test_dataset


def build_training_arguments(config: TrainConfig, device: torch.device) -> TrainingArguments:
    use_cuda = device.type == "cuda"
    bf16_supported = use_cuda and torch.cuda.is_bf16_supported()
    fp16_enabled = use_cuda and not bf16_supported
    dataloader_workers = min(4, os.cpu_count() or 1)

    optim_name = "adamw_torch_fused" if use_cuda else "adamw_torch"

    return TrainingArguments(
        output_dir=config.output_dir,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        save_strategy="epoch",
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=10,
        logging_dir=config.logging_dir,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=1,
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=use_cuda,
        group_by_length=True,
        fp16=fp16_enabled,
        bf16=bf16_supported,
        optim=optim_name,
        report_to="none",
    )


def main():
    config = TrainConfig()
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    files = collect_audio_files(DATA_PATH)
    labels, labels_encoded, label_encoder, class_counts = encode_labels(files)
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
    model = build_model_from_label_encoder(label_encoder, model_name=MODEL_NAME)
    if config.freeze_feature_encoder:
        model.freeze_feature_encoder()
    train_dataset, test_dataset = build_datasets(files, labels_encoded, feature_extractor)

    print(f"Training device: {device}")
    print(f"Found {len(files)} audio files across {len(label_encoder.classes_)} classes.")
    print(dict(zip(label_encoder.classes_, class_counts.tolist())))
    print(
        "Fine-tuning setup:",
        {
            "model_name": MODEL_NAME,
            "freeze_feature_encoder": config.freeze_feature_encoder,
            "train_batch_size": config.train_batch_size,
            "eval_batch_size": config.eval_batch_size,
            "epochs": config.num_train_epochs,
            "learning_rate": config.learning_rate,
        },
    )

    training_args = build_training_arguments(config, device)

    data_collator = DataCollatorWithPadding(feature_extractor)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(trainer.evaluate())
    predictions = trainer.predict(test_dataset)
    print(predictions.metrics)


if __name__ == "__main__":
    main()
