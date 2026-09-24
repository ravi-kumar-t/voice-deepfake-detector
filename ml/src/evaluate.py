"""
Model Evaluation Script for Voice Deepfake Detection Suite.

Evaluates the saved PyTorch CNN-LSTM checkpoint on the validation split:
- Confusion Matrix (TP, TN, FP, FN)
- Overall Accuracy, Precision, Recall, F1 Score
- Per-class metrics for REAL (0) and FAKE (1)
- Sample inference demonstration on individual REAL and FAKE validation clips
"""

from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config
from dataset import get_dataloaders
from model import VoiceDeepfakeCNNLSTM
from predict import predict_audio


def load_trained_model(checkpoint_path: Path = config.MODEL_SAVE_PATH, device: torch.device = config.DEVICE) -> VoiceDeepfakeCNNLSTM:
    """
    Loads model architecture and restores weights from checkpoint.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    model = VoiceDeepfakeCNNLSTM().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch = checkpoint.get("epoch", "N/A")
        val_acc = checkpoint.get("val_acc", "N/A")
        print(f"[Model] Successfully loaded checkpoint from {checkpoint_path.name} (Trained Epoch: {epoch}, Val Acc: {val_acc:.2f}%)")
    else:
        model.load_state_dict(checkpoint)
        print(f"[Model] Successfully loaded checkpoint weights from {checkpoint_path.name}")

    model.eval()
    return model


def evaluate():
    print("=" * 65)
    print("Voice Deepfake Detector: Model Evaluation on Validation Set")
    print(f"Device: {config.DEVICE}")
    print("=" * 65)

    # 1. Load Data with identical seed and split as training
    _, val_loader = get_dataloaders(
        real_dir=config.REAL_DATA_DIR,
        fake_dir=config.FAKE_DATA_DIR,
        max_samples_per_class=config.MAX_SAMPLES_PER_CLASS,
        val_split=config.VAL_SPLIT,
        batch_size=config.BATCH_SIZE,
        random_seed=config.RANDOM_SEED
    )

    if val_loader is None:
        print("[Error] Validation loader could not be created. Please verify data directory.")
        return

    # 2. Load trained model
    model = load_trained_model(config.MODEL_SAVE_PATH, config.DEVICE)

    # 3. Collect predictions and targets
    all_targets = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(config.DEVICE)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs >= 0.5).astype(int)

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy().flatten().astype(int))

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 4. Calculate metrics (Positive class = 1 / FAKE, Negative class = 0 / REAL)
    # Confusion matrix layout: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(all_targets, all_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    acc = accuracy_score(all_targets, all_preds) * 100.0
    prec = precision_score(all_targets, all_preds, zero_division=0) * 100.0
    rec = recall_score(all_targets, all_preds, zero_division=0) * 100.0
    f1 = f1_score(all_targets, all_preds, zero_division=0) * 100.0

    # Per-class metrics
    prec_real = precision_score(all_targets, all_preds, pos_label=0, zero_division=0) * 100.0
    rec_real = recall_score(all_targets, all_preds, pos_label=0, zero_division=0) * 100.0
    f1_real = f1_score(all_targets, all_preds, pos_label=0, zero_division=0) * 100.0

    prec_fake = precision_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100.0
    rec_fake = recall_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100.0
    f1_fake = f1_score(all_targets, all_preds, pos_label=1, zero_division=0) * 100.0

    # 5. Print Results
    print("\n" + "=" * 45)
    print("       VALIDATION METRICS SUMMARY")
    print("=" * 45)
    print(f"Total Validation Samples: {len(all_targets)}")
    print(f"Accuracy:  {acc:.2f}%")
    print(f"Precision: {prec:.2f}%")
    print(f"Recall:    {rec:.2f}%")
    print(f"F1 Score:  {f1:.2f}%")

    print("\n" + "-" * 45)
    print("Confusion Matrix:")
    print(f"  True Positives  (TP - FAKE correctly predicted):  {tp}")
    print(f"  True Negatives  (TN - REAL correctly predicted):  {tn}")
    print(f"  False Positives (FP - REAL predicted as FAKE):   {fp}")
    print(f"  False Negatives (FN - FAKE predicted as REAL):   {fn}")
    print(f"\nMatrix [[TN, FP], [FN, TP]]:\n{cm}")

    print("\n" + "-" * 45)
    print("Per-Class Metrics:")
    print("REAL (Class 0):")
    print(f"  - Precision: {prec_real:.2f}%")
    print(f"  - Recall:    {rec_real:.2f}%")
    print(f"  - F1 Score:  {f1_real:.2f}%")
    print("FAKE (Class 1):")
    print(f"  - Precision: {prec_fake:.2f}%")
    print(f"  - Recall:    {rec_fake:.2f}%")
    print(f"  - F1 Score:  {f1_fake:.2f}%")

    # 6. Sample inference on 2 validation examples (one REAL, one FAKE)
    val_dataset = val_loader.dataset
    print("\n" + "=" * 45)
    print("     SAMPLE VALIDATION INFERENCES")
    print("=" * 45)

    real_sample_path = None
    fake_sample_path = None

    # Find sample paths from the underlying dataset
    for idx in val_dataset.indices:
        fpath = val_dataset.dataset.file_paths[idx]
        flabel = val_dataset.dataset.labels[idx]
        if flabel == 0.0 and real_sample_path is None:
            real_sample_path = fpath
        elif flabel == 1.0 and fake_sample_path is None:
            fake_sample_path = fpath
        if real_sample_path and fake_sample_path:
            break

    # Run inference for REAL sample
    if real_sample_path:
        pred_label, conf, prob = predict_audio(real_sample_path, model, config.DEVICE)
        print(f"\n[Example 1 - REAL]")
        print(f"File:                  {real_sample_path.name}")
        print(f"Actual:                REAL (0)")
        print(f"Predicted:             {pred_label}")
        print(f"Synthetic probability: {prob:.4f}")
        print(f"Confidence:            {conf:.2f}%")

    # Run inference for FAKE sample
    if fake_sample_path:
        pred_label, conf, prob = predict_audio(fake_sample_path, model, config.DEVICE)
        print(f"\n[Example 2 - FAKE]")
        print(f"File:                  {fake_sample_path.name}")
        print(f"Actual:                FAKE (1)")
        print(f"Predicted:             {pred_label}")
        print(f"Synthetic probability: {prob:.4f}")
        print(f"Confidence:            {conf:.2f}%")

    print("\n" + "=" * 45)


if __name__ == "__main__":
    evaluate()
