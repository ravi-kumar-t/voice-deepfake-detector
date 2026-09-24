/**
 * Audio Helper & Digital Signal Processing utilities for Web Audio API
 */

/**
 * Decode an ArrayBuffer or File into an AudioBuffer using an AudioContext
 */
export async function decodeAudioData(fileOrBuffer) {
  const AudioContextClass = window.AudioContext || window.webkitAudioContext;
  const ctx = new AudioContextClass();
  
  let arrayBuffer;
  if (fileOrBuffer instanceof Blob || fileOrBuffer instanceof File) {
    arrayBuffer = await fileOrBuffer.arrayBuffer();
  } else {
    arrayBuffer = fileOrBuffer;
  }

  try {
    const audioBuffer = await ctx.decodeAudioData(arrayBuffer.slice(0));
    await ctx.close();
    return audioBuffer;
  } catch (err) {
    await ctx.close();
    throw new Error(`Failed to decode audio data: ${err.message || err}`);
  }
}

/**
 * Extract peak waveform data for canvas rendering (min/max pairs)
 */
export function extractWaveformPeaks(audioBuffer, numBuckets = 300) {
  const channelData = audioBuffer.getChannelData(0);
  const totalSamples = channelData.length;
  const bucketSize = Math.floor(totalSamples / numBuckets);
  const peaks = [];

  for (let i = 0; i < numBuckets; i++) {
    const start = i * bucketSize;
    const end = Math.min(start + bucketSize, totalSamples);
    let min = 0;
    let max = 0;

    for (let j = start; j < end; j++) {
      const val = channelData[j];
      if (val < min) min = val;
      if (val > max) max = val;
    }

    peaks.push({ min, max, avg: (Math.abs(min) + Math.abs(max)) / 2 });
  }

  return peaks;
}

/**
 * Compute real STFT spectrogram matrix from AudioBuffer
 * Returns a 2D matrix: [timeFrames][freqBins] with normalized dB values (0.0 to 1.0)
 */
export function computeSpectrogram(audioBuffer, fftSize = 512, hopSize = 128) {
  const channelData = audioBuffer.getChannelData(0);
  const sampleRate = audioBuffer.sampleRate;
  const numSamples = channelData.length;
  const numBins = fftSize / 2; // e.g. 256 frequency bins
  const numFrames = Math.floor((numSamples - fftSize) / hopSize);

  if (numFrames <= 0) {
    return { frames: [], maxFreq: sampleRate / 2, sampleRate, duration: audioBuffer.duration };
  }

  // Precompute Hann window
  const window = new Float32Array(fftSize);
  for (let i = 0; i < fftSize; i++) {
    window[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (fftSize - 1)));
  }

  const frames = [];
  let globalMax = -Infinity;
  let globalMin = Infinity;

  // Real & Imag arrays for FFT
  const real = new Float32Array(fftSize);
  const imag = new Float32Array(fftSize);

  for (let f = 0; f < numFrames; f++) {
    const offset = f * hopSize;

    // Apply window
    for (let i = 0; i < fftSize; i++) {
      real[i] = channelData[offset + i] * window[i];
      imag[i] = 0;
    }

    // In-place Cooley-Tukey Radix-2 FFT
    cooleyTukeyFFT(real, imag);

    // Compute magnitude & convert to dB
    const frameMagnitudes = new Float32Array(numBins);
    for (let k = 0; k < numBins; k++) {
      const mag = Math.sqrt(real[k] * real[k] + imag[k] * imag[k]);
      // Convert to dB scale with safety floor
      const db = 20 * Math.log10(Math.max(mag, 1e-6));
      frameMagnitudes[k] = db;
      if (db > globalMax) globalMax = db;
      if (db < globalMin) globalMin = db;
    }

    frames.push(frameMagnitudes);
  }

  // Normalize frame values to [0, 1] for heatmap coloring
  const range = Math.max(globalMax - globalMin, 1e-5);
  const normalizedFrames = frames.map(frame => {
    const norm = new Float32Array(numBins);
    for (let k = 0; k < numBins; k++) {
      norm[k] = Math.max(0, Math.min(1, (frame[k] - globalMin) / range));
    }
    return norm;
  });

  return {
    frames: normalizedFrames,
    numBins,
    numFrames,
    maxFreq: sampleRate / 2,
    sampleRate,
    duration: audioBuffer.duration
  };
}

/**
 * Standard In-place Cooley-Tukey Radix-2 FFT algorithm
 */
function cooleyTukeyFFT(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 0; i < n - 1; i++) {
    if (i < j) {
      let tempR = real[i];
      real[i] = real[j];
      real[j] = tempR;
      let tempI = imag[i];
      imag[i] = imag[j];
      imag[j] = tempI;
    }
    let k = n >> 1;
    while (k <= j) {
      j -= k;
      k >>= 1;
    }
    j += k;
  }

  for (let len = 2; len <= n; len <<= 1) {
    const halfLen = len >> 1;
    const angle = (-2 * Math.PI) / len;
    const wStepR = Math.cos(angle);
    const wStepI = Math.sin(angle);

    for (let i = 0; i < n; i += len) {
      let wR = 1.0;
      let wI = 0.0;
      for (let k = 0; k < halfLen; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + halfLen] * wR - imag[i + k + halfLen] * wI;
        const vI = real[i + k + halfLen] * wI + imag[i + k + halfLen] * wR;

        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + halfLen] = uR - vR;
        imag[i + k + halfLen] = uI - vI;

        const nextWR = wR * wStepR - wI * wStepI;
        const nextWI = wR * wStepI + wI * wStepR;
        wR = nextWR;
        wI = nextWI;
      }
    }
  }
}

/**
 * Plasma / Cyberpunk / Forensic colormap mapping [0, 1] to RGB hex or css
 */
export function getSpectrogramColor(val) {
  // val is 0.0 to 1.0
  // Color scale: Deep Navy (#080c1e) -> Purple (#4c1d95) -> Magenta (#db2777) -> Orange (#f97316) -> Cyan/White (#38bdf8 / #ffffff)
  if (val < 0.2) {
    const t = val / 0.2;
    return `rgb(${Math.round(8 + t * 40)}, ${Math.round(12 + t * 15)}, ${Math.round(30 + t * 90)})`;
  } else if (val < 0.45) {
    const t = (val - 0.2) / 0.25;
    return `rgb(${Math.round(48 + t * 140)}, ${Math.round(27 + t * 12)}, ${Math.round(120 + t * 25)})`;
  } else if (val < 0.7) {
    const t = (val - 0.45) / 0.25;
    return `rgb(${Math.round(188 + t * 60)}, ${Math.round(39 + t * 76)}, ${Math.round(145 - t * 120)})`;
  } else if (val < 0.9) {
    const t = (val - 0.7) / 0.2;
    return `rgb(${Math.round(248 + t * 7)}, ${Math.round(115 + t * 100)}, ${Math.round(25 + t * 100)})`;
  } else {
    const t = (val - 0.9) / 0.1;
    return `rgb(${Math.round(255)}, ${Math.round(215 + t * 40)}, ${Math.round(125 + t * 130)})`;
  }
}
