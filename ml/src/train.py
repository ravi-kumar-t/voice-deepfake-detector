"""
Training script for the Voice Deepfake CNN-LSTM Classifier.

Features:
- Binary Cross Entropy with Logits Loss (nn.BCEWithLogitsLoss)
- Adam Optimizer with Learning Rate Scheduler
- Training & Validation loss and binary accuracy reporting
- Checkpointing: Saves best performing model to ml/models/cnn_lstm_deepfake.pth
- Automatic GPU (CUDA) or CPU device placement
"""

import argparse
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config
from dataset import get_dataloaders
from model import VoiceDeepfakeCNNLSTM


def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """
    Computes binary classification accuracy given raw logits and ground-truth targets.
    """
    probs = torch.sigmoid(logits)
    predictions = (probs >= threshold).float()
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return (correct / total) if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> tuple:
    """
    Runs a single training epoch across all training batches.
    """
    model.train()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)

        loss.backward()
        # Gradient clipping to prevent exploding gradients in LSTM
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (preds == targets).sum().item()
        total_samples += batch_size

    epoch_loss = running_loss / total_samples if total_samples > 0 else 0.0
    epoch_acc = (running_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return epoch_loss, epoch_acc


def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> tuple:
    """
    Evaluates the model on validation batches.
    """
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)
            loss = criterion(logits, targets)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            running_correct += (preds == targets).sum().item()
            total_samples += batch_size

    val_loss = running_loss / total_samples if total_samples > 0 else 0.0
    val_acc = (running_correct / total_samples) * 100.0 if total_samples > 0 else 0.0
    return val_loss, val_acc


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Voice Deepfake Detector")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=config.VAL_SPLIT, help="Validation set split ratio")
    parser.add_argument("--max-samples", type=int, default=config.MAX_SAMPLES_PER_CLASS, help="Max samples per class (default: 100)")
    args = parser.parse_args()

    print("=" * 65)
    print("Voice Deepfake Detector: CNN-LSTM Training Pipeline")
    print(f"Device in use: {config.DEVICE}")
    print("=" * 65)

    # 1. Load Data
    train_loader, val_loader = get_dataloaders(
        real_dir=config.REAL_DATA_DIR,
        fake_dir=config.FAKE_DATA_DIR,
        max_samples_per_class=args.max_samples,
        val_split=args.val_split,
        batch_size=args.batch_size
    )

    if train_loader is None or val_loader is None:
        print("\n[Notice] Cannot start training because no dataset audio files were found.")
        print("Please place authentic WAV/MP3 files into:")
        print(f"  -> {config.REAL_DATA_DIR}")
        print("And synthetic/AI-generated WAV/MP3 files into:")
        print(f"  -> {config.FAKE_DATA_DIR}")
        print("Then run 'python src/train.py' again.")
        return

    # 2. Instantiate Model, Loss Function, and Optimizer
    model = VoiceDeepfakeCNNLSTM().to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    # Ensure models directory exists
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    start_time = time.time()

    print(f"\nStarting training for {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, config.DEVICE)

        scheduler.step(val_loss)
        elapsed = time.time() - epoch_start

        # Check for improvement and save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": {
                    "n_mels": config.N_MELS,
                    "sample_rate": config.SAMPLE_RATE,
                    "audio_duration": config.AUDIO_DURATION
                }
            }, config.MODEL_SAVE_PATH)
            save_marker = " [Saved Best Checkpoint]"
        else:
            save_marker = ""

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] ({elapsed:.1f}s) | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:5.2f}% | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:5.2f}%{save_marker}"
        )

    total_time = time.time() - start_time
    print("\n" + "=" * 65)
    print(f"Training completed in {total_time:.1f}s")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.2f}%")
    print(f"Model saved to: {config.MODEL_SAVE_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
