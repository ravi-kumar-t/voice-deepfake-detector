"""
PyTorch CNN-LSTM Architecture for Voice Deepfake Detection.

Architecture Overview:
1. 2D CNN Feature Extractor:
   - Input: Mel-Spectrogram (Batch, 1, n_mels=80, time_steps)
   - Conv2D (1 -> 32) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
   - Conv2D (32 -> 64) -> BatchNorm -> ReLU -> MaxPool2D(2, 2)
   - Output: Feature maps of shape (Batch, 64, 20, time_steps / 4)

2. Temporal Sequence Reshaping:
   - Permute & Flatten frequency and channel dimensions: (Batch, time_steps / 4, 64 * 20)
   - Input dimension per time-step: 1280 features

3. Temporal Modeling (LSTM):
   - Multi-layer Bidirectional LSTM over time steps
   - Captures temporal inconsistencies, glitching, and unnatural prosody

4. Classifier Head:
   - Global Temporal Average Pooling
   - Fully Connected Layers -> Single Raw Logit (Binary Classification)
"""

from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent))
import config


class VoiceDeepfakeCNNLSTM(nn.Module):
    """
    CNN-LSTM Deepfake Audio Detection Model.
    """
    def __init__(
        self,
        n_mels: int = config.N_MELS,
        cnn_channels: list = config.CNN_CHANNELS,
        lstm_hidden: int = config.LSTM_HIDDEN_SIZE,
        lstm_layers: int = config.LSTM_NUM_LAYERS,
        bidirectional: bool = config.LSTM_BIDIRECTIONAL,
        dropout: float = config.DROPOUT_RATE
    ):
        super(VoiceDeepfakeCNNLSTM, self).__init__()

        # ---------------------------------------------------------------------
        # 1. 2D CNN Feature Extractor (Spectral Spatial Analysis)
        # ---------------------------------------------------------------------
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=cnn_channels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces (80, T) -> (40, T/2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=cnn_channels[0], out_channels=cnn_channels[1], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(cnn_channels[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Reduces (40, T/2) -> (20, T/4)
        )

        # Calculate flattened feature dimension per time frame after 2 MaxPool2D(2, 2) operations
        reduced_freq_bins = n_mels // 4  # 80 // 4 = 20
        self.lstm_input_dim = cnn_channels[1] * reduced_freq_bins  # 64 * 20 = 1280

        # ---------------------------------------------------------------------
        # 2. LSTM Temporal Sequence Modeling
        # ---------------------------------------------------------------------
        self.lstm = nn.LSTM(
            input_size=self.lstm_input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        lstm_output_dim = lstm_hidden * (2 if bidirectional else 1)

        # ---------------------------------------------------------------------
        # 3. Binary Classification Head
        # ---------------------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Returns single raw logit
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Mel-spectrogram tensor of shape (Batch, 1, N_MELS, Time_Steps)
               or (Batch, N_MELS, Time_Steps)
               
        Returns:
            logit: Raw unnormalized logit of shape (Batch, 1)
        """
        # Ensure 4D tensor: (Batch, Channel=1, Freq=80, Time)
        if x.dim() == 3:
            x = x.unsqueeze(1)

        # CNN Feature Extraction
        # x: (B, 1, 80, T) -> (B, 32, 40, T/2)
        x = self.conv_block1(x)
        # x: (B, 32, 40, T/2) -> (B, 64, 20, T/4)
        x = self.conv_block2(x)

        # Reshape for LSTM: (Batch, Channels, Freq, Time) -> (Batch, Time, Channels * Freq)
        # Permute to (Batch, Time, Channels, Freq)
        batch_size, channels, freq, time_steps = x.shape
        x = x.permute(0, 3, 1, 2)  # (B, T', C, F)
        x = x.contiguous().view(batch_size, time_steps, channels * freq)  # (B, T', 1280)

        # LSTM Temporal Modeling
        # lstm_out: (B, T', lstm_output_dim)
        lstm_out, _ = self.lstm(x)

        # Temporal Global Average Pooling across time dimension
        pooled = torch.mean(lstm_out, dim=1)  # (B, lstm_output_dim)

        # Binary Classifier Head
        logit = self.classifier(pooled)  # (B, 1)

        return logit


if __name__ == "__main__":
    # Self-test the forward pass with dummy tensor for 2.0s audio (126 frames)
    test_input = torch.randn(2, 1, config.N_MELS, 126)
    model = VoiceDeepfakeCNNLSTM()
    output = model(test_input)
    print("VoiceDeepfakeCNNLSTM Model Architecture:")
    print(model)
    print(f"\nForward Pass Test: Input shape {test_input.shape} -> Output shape {output.shape}")
    print("Model initialized and verified successfully!")
