"""
Inference script for the Voice Deepfake CNN-LSTM Detector.

Usage:
  python src/predict.py path/to/audio.wav
  python src/predict.py --audio path/to/sample.mp3 --model path/to/checkpoint.pth

Output Format:
  Prediction: REAL or SYNTHETIC
  Confidence: XX.XX%
"""

import argparse
from pathlib import Path
import torch

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config
from preprocessing import (
    preprocess_audio,
    load_audio,
    convert_to_mono,
    resample_audio,
    normalize_waveform,
    pad_or_truncate,
    extract_mel_spectrogram
)
from model import VoiceDeepfakeCNNLSTM


def load_model(checkpoint_path: Path = config.MODEL_SAVE_PATH, device: torch.device = config.DEVICE) -> VoiceDeepfakeCNNLSTM:
    """
    Loads model architecture and loads saved weights checkpoint if present.
    """
    model = VoiceDeepfakeCNNLSTM().to(device)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.eval()
        return model
    else:
        raise FileNotFoundError(
            f"Trained model checkpoint not found at '{checkpoint_path}'.\n"
            f"Please run 'python src/train.py' first after adding training data to ml/data/."
        )


def predict_audio(
    audio_path: Path,
    model: VoiceDeepfakeCNNLSTM,
    device: torch.device = config.DEVICE
) -> tuple:
    """
    Runs deepfake prediction on an audio file.
    
    Args:
        audio_path: Path to the target audio file.
        model: Loaded VoiceDeepfakeCNNLSTM model.
        device: CPU or CUDA device.
        
    Returns:
        label (str): "REAL" or "SYNTHETIC"
        confidence (float): Confidence percentage (0.00 to 100.00)
        prob_fake (float): Probability score of being fake/synthetic (0.0 to 1.0)
    """
    # 1. Preprocess audio to Mel-spectrogram tensor: Shape (1, 80, 251)
    mel_tensor = preprocess_audio(audio_path)
    
    # 2. Add batch dimension: Shape (1, 1, 80, 251)
    mel_tensor = mel_tensor.unsqueeze(0).to(device)

    # 3. Model inference
    model.eval()
    with torch.no_grad():
        logit = model(mel_tensor)
        prob_fake = torch.sigmoid(logit).item()

    # 4. Determine label and confidence
    # Label 1.0 is SYNTHETIC/FAKE, Label 0.0 is REAL
    if prob_fake >= 0.5:
        prediction = "SYNTHETIC"
        confidence = prob_fake * 100.0
    else:
        prediction = "REAL"
        confidence = (1.0 - prob_fake) * 100.0

    return prediction, confidence, prob_fake


def predict_audio_segments(
    audio_path: Path,
    model: VoiceDeepfakeCNNLSTM,
    device: torch.device = config.DEVICE,
    num_segments: int = 4
) -> list:
    """
    Performs segment-level deepfake analysis using native temporal slicing.
    
    1. Passes full audio through the standard preprocessing to produce an 80-bin Mel-spectrogram.
    2. Runs a single forward pass through CNN feature extraction and BiLSTM temporal sequence modeling.
    3. Divides the resulting BiLSTM temporal hidden states into consecutive temporal slices.
    4. Mean-pools each segment's temporal representations and passes them through the classifier head.
    5. Applies sigmoid activation to compute localized segment evidence probabilities.
    """
    # 1. Preprocess full audio using existing pipeline: Shape (1, 80, Time_Steps)
    mel_tensor = preprocess_audio(audio_path)

    # 2. Add batch dimension and move to device: Shape (1, 1, 80, Time_Steps)
    mel_tensor = mel_tensor.unsqueeze(0).to(device)

    segments = []
    model.eval()

    with torch.no_grad():
        # CNN Feature Extraction
        x = model.conv_block1(mel_tensor)
        x = model.conv_block2(x)

        # Reshape for LSTM: (Batch, Time, Channels * Freq)
        batch_size, channels, freq, time_steps = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, time_steps, channels * freq)

        # BiLSTM Temporal Sequence Modeling: Shape (Batch, Time, LSTM_Dim)
        lstm_out, _ = model.lstm(x)

        # Dynamically partition temporal frames into consecutive regions
        t = time_steps
        seg_frames = t / float(num_segments)

        for i in range(num_segments):
            start_sec = i * (config.AUDIO_DURATION / num_segments)
            end_sec = (i + 1) * (config.AUDIO_DURATION / num_segments)

            s_idx = int(round(i * seg_frames))
            e_idx = int(round((i + 1) * seg_frames))
            if s_idx >= e_idx:
                e_idx = s_idx + 1
            e_idx = min(e_idx, t)

            # Mean-pool LSTM temporal representations for this specific time window
            seg_pooled = torch.mean(lstm_out[:, s_idx:e_idx, :], dim=1)

            # Pass through existing classifier head
            seg_logit = model.classifier(seg_pooled)
            prob_fake = torch.sigmoid(seg_logit).item()

            if prob_fake >= 0.5:
                pred = "SYNTHETIC"
                conf = prob_fake * 100.0
            else:
                pred = "REAL"
                conf = (1.0 - prob_fake) * 100.0

            segments.append({
                "start": round(start_sec, 2),
                "end": round(end_sec, 2),
                "prediction": pred,
                "synthetic_probability": round(prob_fake, 4),
                "confidence": round(conf, 2)
            })

    return segments


def main():
    parser = argparse.ArgumentParser(description="Voice Deepfake Detector - Audio Inference")
    parser.add_argument("audio_path", type=str, nargs="?", default=None, help="Path to input audio file (.wav, .mp3, etc.)")
    parser.add_argument("--audio", "-a", type=str, dest="opt_audio", default=None, help="Alternative flag for audio filepath")
    parser.add_argument("--model", "-m", type=str, default=str(config.MODEL_SAVE_PATH), help="Path to model weights checkpoint")
    args = parser.parse_args()

    target_audio = args.audio_path or args.opt_audio

    if not target_audio:
        print("Error: Please provide an audio filepath to analyze.")
        print("Example: python src/predict.py path/to/sample.wav")
        return

    audio_file = Path(target_audio)
    if not audio_file.exists():
        print(f"Error: Audio file not found at '{audio_file}'.")
        return

    model_path = Path(args.model)

    try:
        model = load_model(model_path, config.DEVICE)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    prediction, confidence, prob_fake = predict_audio(audio_file, model, config.DEVICE)

    # Output formatted prediction
    print("=" * 40)
    print(f"File: {audio_file.name}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"Raw Synthetic Probability: {prob_fake:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
