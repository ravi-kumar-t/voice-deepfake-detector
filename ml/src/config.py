"""
Configuration settings for the Voice Deepfake Detection ML pipeline.
Designed for 16 kHz mono audio processing, Mel-spectrogram extraction,
and CNN-LSTM binary classification.
"""

from pathlib import Path
import torch

# -----------------------------------------------------------------------------
# Directory and File Paths
# -----------------------------------------------------------------------------
SRC_DIR = Path(__file__).resolve().parent
ML_DIR = SRC_DIR.parent
DATA_DIR = ML_DIR / "data"
REAL_DATA_DIR = DATA_DIR / "real"
FAKE_DATA_DIR = DATA_DIR / "fake"
MODELS_DIR = ML_DIR / "models"
MODEL_SAVE_PATH = MODELS_DIR / "cnn_lstm_deepfake.pth"

# Supported audio extensions
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# -----------------------------------------------------------------------------
# Audio & Spectrogram Parameters
# -----------------------------------------------------------------------------
SAMPLE_RATE = 16000        # Target sample rate (16 kHz mono)
AUDIO_DURATION = 2.0       # Target duration in seconds (2 seconds for FOR-2sec)
TARGET_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION)  # 32,000 samples

N_MELS = 80                # Number of Mel frequency bands
N_FFT = 1024               # FFT window size
HOP_LENGTH = 256           # Hop length for STFT
F_MIN = 0.0                # Minimum frequency for Mel filterbank
F_MAX = SAMPLE_RATE / 2    # Maximum frequency (Nyquist = 8000 Hz)

# -----------------------------------------------------------------------------
# Model Architecture Parameters
# -----------------------------------------------------------------------------
CNN_CHANNELS = [32, 64]    # Output channels for Conv2D layers
LSTM_HIDDEN_SIZE = 64      # Hidden units per LSTM direction
LSTM_NUM_LAYERS = 2        # Stacked LSTM layers
LSTM_BIDIRECTIONAL = True  # Bidirectional LSTM
DROPOUT_RATE = 0.3         # Dropout probability

# -----------------------------------------------------------------------------
# Training Hyperparameters
# -----------------------------------------------------------------------------
BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2
RANDOM_SEED = 42
MAX_SAMPLES_PER_CLASS = 100  # Balanced subset limit (100 real + 100 fake)

# Device selection: use CUDA if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
