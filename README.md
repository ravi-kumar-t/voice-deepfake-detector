# VoiceGuard AI

> **Generative Voice & Audio Deepfake Detection Suite**

VoiceGuard AI is a deep learning audio forensics suite designed to detect synthetic and manipulated speech. By combining 2D Convolutional Neural Networks (CNN) on high-resolution Mel-spectrograms with Bidirectional Long Short-Term Memory (BiLSTM) recurrent networks, VoiceGuard AI evaluates both frequency-domain acoustic anomalies and temporal phoneme transitions to differentiate authentic human speech from AI-generated audio.

---

## 1. Overview

With the rapid proliferation of generative text-to-speech (TTS) and voice conversion models, synthetic speech deepfakes present critical challenges to biometric security, identity verification, and media integrity. 

VoiceGuard AI processes voice audio recordings and predicts whether a given clip is **REAL** (authentic human) or **SYNTHETIC** (AI-generated / manipulated). The system couples an optimized PyTorch CNN-BiLSTM inference engine (~10–15 ms latency per clip) with a cybersecurity-grade React dashboard featuring real-time Web Audio API waveform and spectrogram visualizers.

---

## 2. Features

- **Drag-and-Drop Audio Ingestion**: Native drag-and-drop and file browser for 16 kHz `.wav` audio files.
- **In-Browser Microphone Recording**: Record voice samples directly in the browser with automatic client-side 16-bit linear PCM WAV encoding.
- **Interactive Waveform Visualizer**: Real-time canvas rendering of peak amplitude and zero-crossing dynamics.
- **STFT Spectral Heatmap (0–8 kHz)**: Real-time FFT time-frequency spectrogram decomposition computed directly from the audio buffer.
- **CNN-BiLSTM Deepfake Detection**: High-speed neural inference analyzing spectral patterns and temporal transitions.
- **Confidence & Probability Metrics**: Clear classification verdict, confidence percentage, and synthetic probability bar with a 50% decision threshold.
- **Segment-Level Acoustic Breakdown**: 500ms time-sliced energy and stability analysis across four consecutive windows.
- **FastAPI Backend**: Asynchronous REST API providing health diagnostics and multipart audio prediction endpoints.
- **Modern Forensics Dashboard**: Cyber-defense dark mode UI built with React, Vite, and glassmorphic styling.

---

## 3. Architecture

```text
Audio
  ↓
16 kHz Mono
  ↓
2-second window
  ↓
Mel Spectrogram
  ↓
2D CNN
  ↓
BiLSTM
  ↓
Binary Classifier
  ↓
REAL / SYNTHETIC
```

---

## 4. Model Pipeline

The end-to-end machine learning pipeline operates as follows:

1. **Audio Standardization**:
   - Audio is resampled to **16,000 Hz mono**.
   - Amplitude is normalized to `[-1.0, 1.0]` to prevent gain bias.
   - Analysis window is fixed to **2.0 seconds** (**32,000 samples**), using centered padding or truncation.
2. **Spectral Feature Extraction**:
   - **80-bin Mel Spectrogram** computed using STFT (`N_FFT = 1024`, `Hop Length = 256`, `F_MIN = 0 Hz`, `F_MAX = 8000 Hz`).
   - Log-power scaling (dB) followed by standard zero-mean unit-variance z-score normalization.
   - Output feature shape: `(1, 80, 126)`.
3. **Spatial Feature Learning (2D CNN)**:
   - Two convolutional blocks with batch normalization, ReLU activation, and MaxPooling extract local spectral textures and harmonic patterns.
4. **Temporal Modeling (BiLSTM)**:
   - 2-layer Bidirectional LSTM (64 hidden units per direction) tracks sequential frequency dynamics and vocoder artifacts over time.
5. **Classification**:
   - Global temporal average pooling followed by fully-connected linear layers with dropout (0.3) produces a single binary logit mapped through a sigmoid activation to output synthetic probability.

---

## 5. Tech Stack

- **Frontend**:
  - React 19
  - Vite
  - Web Audio API (Canvas STFT & Waveform Rendering)
  - Lucide React
- **Backend**:
  - Python 3.11 / 3.12 / 3.13
  - FastAPI
  - Uvicorn
  - SoundFile & Librosa
- **Machine Learning**:
  - PyTorch
  - Torchaudio
  - 2D CNN + Bidirectional LSTM
  - 80-bin Mel-filterbank Spectrograms

---

## 6. Project Structure

```text
voice-deepfake-detector/
├── .gitignore                      # Root repository ignores (virtualenvs, node_modules, data)
├── README.md                       # Documentation & setup guide
├── frontend/
│   ├── index.html                  # HTML entry with Google Fonts & metadata
│   ├── package.json                # React & Vite dependencies
│   ├── vite.config.js              # Vite server configuration
│   ├── .env.example                # Environment template for frontend API URL
│   ├── public/
│   │   └── samples/                # Demo test samples (real_0001.wav, real_0002.wav, fake_0001.wav)
│   └── src/
│       ├── App.jsx                 # Master UI state & health polling
│       ├── index.css               # Cybersecurity design system & tokens
│       ├── services/
│       │   └── api.js              # FastAPI communication (/health, /predict)
│       ├── utils/
│       │   ├── wavEncoder.js       # Client-side 16-bit PCM WAV encoder
│       │   └── audioHelper.js      # FFT spectrogram & peak extraction
│       └── components/
│           ├── Header.jsx          # Branding & live backend status badge
│           ├── AudioInput.jsx      # Drag-and-drop, mic recording, demo selector
│           ├── AudioPlayer.jsx     # HTML5 / WebAudio player controls
│           ├── DetectionResult.jsx # Verdict card, confidence gauge, probability bar
│           ├── WaveformViewer.jsx  # Interactive peak amplitude canvas visualizer
│           ├── SpectrogramViewer.jsx# Real-time STFT 0–8 kHz heatmap canvas
│           ├── SegmentAnalysis.jsx # 500ms time-sliced RMS energy breakdown
│           ├── ModelSpecsCard.jsx  # Technical architecture specifications
│           └── HowItWorks.jsx      # 3-step forensics methodology
└── ml/
    ├── Procfile                    # Deployment entrypoint for cloud web services
    ├── api/
    │   ├── main.py                 # FastAPI application
    │   └── requirements.txt        # Production API dependencies
    ├── models/
    │   └── cnn_lstm_deepfake.pth   # 9.32 MB trained PyTorch model checkpoint
    └── src/
        ├── config.py               # Pipeline hyperparameters and constants
        ├── preprocessing.py        # Audio loading, resampling, and Mel-spectrogram extraction
        ├── model.py                # PyTorch CNN-BiLSTM neural network definition
        ├── dataset.py              # PyTorch Dataset and train/val split logic
        ├── train.py                # Model training loop and checkpoint saver
        ├── evaluate.py             # Validation metrics calculation script
        └── predict.py              # Standalone CLI inference engine
```

---

## 7. Local Setup

### Prerequisites
- Node.js (v18+ or v20+) and npm
- Python (v3.10, v3.11, v3.12, or v3.13)

### 1. Backend Setup

```bash
cd ml

# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
# python3 -m venv .venv
# source .venv/bin/activate

# Install dependencies
pip install -r api/requirements.txt

# Start FastAPI server
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

The API will be accessible at `http://127.0.0.1:8000` (Interactive docs: `http://127.0.0.1:8000/docs`).

### 2. Frontend Setup

In a separate terminal:

```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev
```

Open `http://127.0.0.1:5173/` in your browser.

---

## 8. Environment Configuration

The frontend dynamically connects to the backend using the `VITE_API_URL` environment variable.

Create a `frontend/.env` file if deploying to a remote host:

```env
VITE_API_URL=http://127.0.0.1:8000
```

If `VITE_API_URL` is omitted, the frontend automatically defaults to `http://127.0.0.1:8000` for local development.

---

## 9. Model Performance

On the current 40-sample held-out validation split from the prototype dataset subset, the model achieved:

- **Accuracy**: 90.00%
- **Precision**: 86.96%
- **Recall**: 95.24%
- **F1 Score**: 90.91%

*These results are prototype validation results and should not be interpreted as general real-world accuracy.*

---

## 10. Limitations

- **Prototype Training Subset**: The current model is trained on a relatively small prototype subset.
- **Domain Shifts**: Performance may vary on unseen recording conditions, speakers, codecs, microphones, and unseen synthesis systems.
- **Probabilistic Evidence**: A prediction is model evidence, not definitive proof of manipulation.
- **Fixed Window**: Current model input is based on 2-second analysis windows.
