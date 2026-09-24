"""
PyTorch Dataset and DataLoader Utilities for Audio Deepfake Detection.

Label Mapping:
  - REAL audio: label 0.0
  - FAKE (Synthetic/Deepfake) audio: label 1.0

Directory Layout:
  ml/data/
    ├── real/   <- Put authentic human audio files (.wav, .mp3, etc.)
    └── fake/   <- Put AI-generated synthetic audio files (.wav, .mp3, etc.)
"""

from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config
from preprocessing import preprocess_audio


class DeepfakeAudioDataset(Dataset):
    """
    Custom PyTorch Dataset for loading and preprocessing real and fake audio samples.
    """
    
    def __init__(self, file_paths: List[Path], labels: List[float]):
        """
        Args:
            file_paths: List of Paths to audio files.
            labels: List of float labels (0.0 for real, 1.0 for fake).
        """
        assert len(file_paths) == len(labels), "File paths and labels count must match."
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            mel_spec: Spectrogram tensor of shape (1, N_MELS, TIME_STEPS)
            label: Tensor of shape (1,) with value 0.0 or 1.0
        """
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        try:
            # Preprocess audio into 80-bin Mel-spectrogram
            mel_spec = preprocess_audio(filepath)
        except Exception as e:
            # Dynamic fallback to zero-filled tensor on corrupt audio file
            expected_time_frames = 1 + (config.TARGET_SAMPLES // config.HOP_LENGTH)
            print(f"[Warning] Failed to load {filepath.name}: {e}. Using silent placeholder of shape (1, {config.N_MELS}, {expected_time_frames}).")
            mel_spec = torch.zeros((1, config.N_MELS, expected_time_frames), dtype=torch.float32)

        label_tensor = torch.tensor([label], dtype=torch.float32)
        return mel_spec, label_tensor


def scan_audio_files(
    real_dir: Path = config.REAL_DATA_DIR,
    fake_dir: Path = config.FAKE_DATA_DIR,
    max_samples_per_class: Optional[int] = config.MAX_SAMPLES_PER_CLASS,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[List[Path], List[float]]:
    """
    Scans real and fake data directories for audio files.
    Deterministically selects a balanced subset up to max_samples_per_class.
    
    Returns:
        file_paths: List of selected audio Paths.
        labels: List of matching float labels (0.0 for real, 1.0 for fake).
    """
    real_paths: List[Path] = []
    fake_paths: List[Path] = []

    # Collect Real audio files (label = 0.0)
    if real_dir.exists():
        for ext in config.AUDIO_EXTENSIONS:
            for file in sorted(real_dir.glob(f"*{ext}")):
                real_paths.append(file)
            for file in sorted(real_dir.glob(f"*{ext.upper()}")):
                real_paths.append(file)

    # Collect Fake audio files (label = 1.0)
    if fake_dir.exists():
        for ext in config.AUDIO_EXTENSIONS:
            for file in sorted(fake_dir.glob(f"*{ext}")):
                fake_paths.append(file)
            for file in sorted(fake_dir.glob(f"*{ext.upper()}")):
                fake_paths.append(file)

    # Deterministic sorting
    real_paths = sorted(list(dict.fromkeys(real_paths)))
    fake_paths = sorted(list(dict.fromkeys(fake_paths)))

    # Apply deterministic cap if specified
    if max_samples_per_class is not None:
        real_paths = real_paths[:max_samples_per_class]
        fake_paths = fake_paths[:max_samples_per_class]

    file_paths = real_paths + fake_paths
    labels = [0.0] * len(real_paths) + [1.0] * len(fake_paths)

    return file_paths, labels


def get_dataloaders(
    real_dir: Path = config.REAL_DATA_DIR,
    fake_dir: Path = config.FAKE_DATA_DIR,
    max_samples_per_class: Optional[int] = config.MAX_SAMPLES_PER_CLASS,
    val_split: float = config.VAL_SPLIT,
    batch_size: int = config.BATCH_SIZE,
    random_seed: int = config.RANDOM_SEED
) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
    """
    Creates train and validation PyTorch DataLoaders with stratified/random split.
    
    Args:
        real_dir: Directory containing real audio files.
        fake_dir: Directory containing fake audio files.
        max_samples_per_class: Maximum number of samples to load per class (default: 100).
        val_split: Fraction of data reserved for validation (e.g., 0.2).
        batch_size: Batch size for DataLoader.
        random_seed: Seed for reproducible train/val splitting.
        
    Returns:
        (train_loader, val_loader) or (None, None) if no samples found.
    """
    file_paths, labels = scan_audio_files(
        real_dir=real_dir,
        fake_dir=fake_dir,
        max_samples_per_class=max_samples_per_class,
        random_seed=random_seed
    )
    total_samples = len(file_paths)

    if total_samples == 0:
        print(f"[Dataset] No audio files found in '{real_dir}' or '{fake_dir}'.")
        print(f"[Dataset] Please add real samples to '{real_dir}' and synthetic samples to '{fake_dir}'.")
        return None, None

    real_count = sum(1 for y in labels if y == 0.0)
    fake_count = sum(1 for y in labels if y == 1.0)

    dataset = DeepfakeAudioDataset(file_paths, labels)

    # Compute split lengths (80% train, 20% validation)
    val_len = int(total_samples * val_split)
    train_len = total_samples - val_len

    if val_len == 0 and total_samples > 1:
        val_len = 1
        train_len = total_samples - 1

    generator = torch.Generator().manual_seed(random_seed)
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0  # Safe default on Windows
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    print(f"[Dataset] REAL samples: {real_count}")
    print(f"[Dataset] FAKE samples: {fake_count}")
    print(f"[Dataset] TOTAL samples: {total_samples}")
    print(f"[Dataset] Train samples: {len(train_dataset)} | Validation samples: {len(val_dataset)}")
    return train_loader, val_loader
