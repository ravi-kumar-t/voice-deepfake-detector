"""
Audio Preprocessing Pipeline for Voice Deepfake Detection.

Pipeline Steps:
1. Load audio file (supports WAV, MP3, FLAC, OGG, etc.)
2. Convert multi-channel audio to mono
3. Resample to 16 kHz
4. Amplitude normalization
5. Pad or truncate to fixed 4-second length (64,000 samples)
6. Compute 80-bin log-scale Mel Spectrogram
7. Return PyTorch tensor ready for CNN-LSTM input: Shape (1, N_MELS, TIME_STEPS)
"""

from pathlib import Path
from typing import Union, Tuple
import torch
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config


# -----------------------------------------------------------------------------
# Pre-instantiated Mel-Spectrogram transform for performance
# -----------------------------------------------------------------------------
mel_transform = T.MelSpectrogram(
    sample_rate=config.SAMPLE_RATE,
    n_fft=config.N_FFT,
    hop_length=config.HOP_LENGTH,
    n_mels=config.N_MELS,
    f_min=config.F_MIN,
    f_max=config.F_MAX,
    power=2.0
)

amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=80)


def load_audio(filepath: Union[str, Path]) -> Tuple[torch.Tensor, int]:
    """
    Loads an audio file and returns the waveform tensor and original sample rate.
    Uses torchaudio with fallback to librosa/soundfile for broad format compatibility.
    
    Args:
        filepath: Path to audio file.
        
    Returns:
        waveform: torch.Tensor of shape (channels, samples)
        sample_rate: int
    """
    filepath = str(filepath)
    try:
        waveform, sr = torchaudio.load(filepath)
        return waveform, sr
    except Exception:
        # Fallback to librosa if torchaudio backend fails on certain formats
        audio_np, sr = librosa.load(filepath, sr=None, mono=False)
        if audio_np.ndim == 1:
            audio_np = audio_np[np.newaxis, :]
        waveform = torch.from_numpy(audio_np).float()
        return waveform, sr


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    """
    Converts multi-channel waveform to single-channel (mono) by averaging channels.
    
    Args:
        waveform: Shape (channels, samples)
        
    Returns:
        mono_waveform: Shape (1, samples)
    """
    if waveform.dim() == 1:
        return waveform.unsqueeze(0)
    if waveform.size(0) > 1:
        # Average across channels
        return torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_audio(waveform: torch.Tensor, orig_sr: int, target_sr: int = config.SAMPLE_RATE) -> torch.Tensor:
    """
    Resamples waveform to target sample rate (16 kHz).
    
    Args:
        waveform: Shape (1, samples)
        orig_sr: Original sample rate
        target_sr: Target sample rate (default: 16000)
        
    Returns:
        resampled_waveform: Shape (1, new_samples)
    """
    if orig_sr == target_sr:
        return waveform
    
    resampler = T.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)


def normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    """
    Normalizes audio waveform amplitude to [-1, 1] range.
    Handles silent or zero-amplitude audio safely.
    
    Args:
        waveform: Shape (1, samples)
        
    Returns:
        normalized_waveform: Shape (1, samples)
    """
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1e-6:
        return waveform / max_val
    return waveform


def pad_or_truncate(waveform: torch.Tensor, target_samples: int = config.TARGET_SAMPLES) -> torch.Tensor:
    """
    Pads with zeros or truncates waveform to fixed length (4 seconds = 64,000 samples).
    
    Args:
        waveform: Shape (1, samples)
        target_samples: Target sample count (default: 64000)
        
    Returns:
        adjusted_waveform: Shape (1, target_samples)
    """
    num_samples = waveform.size(1)
    
    if num_samples > target_samples:
        # Truncate to target length
        return waveform[:, :target_samples]
    elif num_samples < target_samples:
        # Pad with zeros at the end
        padding = target_samples - num_samples
        return torch.nn.functional.pad(waveform, (0, padding), mode="constant", value=0.0)
    return waveform


def extract_mel_spectrogram(waveform: torch.Tensor) -> torch.Tensor:
    """
    Converts a 1D audio waveform into an 80-bin log Mel Spectrogram (in dB).
    Normalizes spectrogram values to zero-mean and unit-variance.
    
    Args:
        waveform: Shape (1, target_samples)
        
    Returns:
        log_mel_spec: Shape (1, N_MELS, TIME_STEPS) e.g., (1, 80, 251)
    """
    # Generate power Mel spectrogram: Shape (1, n_mels, time_steps)
    mel_spec = mel_transform(waveform)
    
    # Convert to decibel (log) scale
    log_mel = amplitude_to_db(mel_spec)
    
    # Normalize spectrogram features (standardization)
    mean = log_mel.mean()
    std = log_mel.std()
    if std > 1e-6:
        log_mel = (log_mel - mean) / std
    else:
        log_mel = log_mel - mean
        
    return log_mel


def preprocess_audio(filepath: Union[str, Path]) -> torch.Tensor:
    """
    Complete end-to-end preprocessing pipeline from file path to model-ready tensor.
    
    Args:
        filepath: Path to audio file.
        
    Returns:
        mel_tensor: torch.Tensor of shape (1, N_MELS, TIME_STEPS)
    """
    # 1. Load audio
    waveform, orig_sr = load_audio(filepath)
    
    # 2. Convert to mono
    mono_waveform = convert_to_mono(waveform)
    
    # 3. Resample to 16 kHz
    resampled_waveform = resample_audio(mono_waveform, orig_sr, config.SAMPLE_RATE)
    
    # 4. Normalize amplitude
    norm_waveform = normalize_waveform(resampled_waveform)
    
    # 5. Pad or truncate to 4 seconds
    fixed_waveform = pad_or_truncate(norm_waveform, config.TARGET_SAMPLES)
    
    # 6. Extract 80-bin Mel Spectrogram
    mel_tensor = extract_mel_spectrogram(fixed_waveform)
    
    return mel_tensor
