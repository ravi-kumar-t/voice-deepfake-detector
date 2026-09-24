/**
 * WAV Encoder Utility
 * Converts Float32Array PCM audio buffer or AudioBuffer into a valid 16-bit Linear PCM WAV Blob.
 */

export function encodeWAV(audioBuffer, targetSampleRate = 16000) {
  const numChannels = 1; // mono
  const sampleRate = targetSampleRate;
  
  // Resample / downmix to mono Float32Array
  const monoSamples = downmixAndResample(audioBuffer, targetSampleRate);
  const numSamples = monoSamples.length;
  
  const byteRate = sampleRate * numChannels * 2; // 16-bit = 2 bytes
  const blockAlign = numChannels * 2;
  const dataSize = numSamples * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  // RIFF chunk descriptor
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(view, 8, 'WAVE');

  // fmt sub-chunk
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true); // Subchunk1Size (16 for PCM)
  view.setUint16(20, 1, true);  // AudioFormat (1 = PCM)
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // BitsPerSample (16-bit)

  // data sub-chunk
  writeString(view, 36, 'data');
  view.setUint32(40, dataSize, true);

  // Write 16-bit PCM samples
  let offset = 44;
  for (let i = 0; i < numSamples; i++) {
    let s = Math.max(-1, Math.min(1, monoSamples[i]));
    // Scale to 16-bit signed integer [-32768, 32767]
    const intSample = s < 0 ? s * 0x8000 : s * 0x7FFF;
    view.setInt16(offset, intSample, true);
    offset += 2;
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view, offset, string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/**
 * Converts any AudioBuffer to mono Float32Array at targetSampleRate
 */
export function downmixAndResample(audioBuffer, targetRate) {
  const origRate = audioBuffer.sampleRate;
  const numChannels = audioBuffer.numberOfChannels;
  const origLength = audioBuffer.length;

  // 1. Downmix to mono
  const mono = new Float32Array(origLength);
  for (let c = 0; c < numChannels; c++) {
    const channelData = audioBuffer.getChannelData(c);
    for (let i = 0; i < origLength; i++) {
      mono[i] += channelData[i] / numChannels;
    }
  }

  // If sample rate matches, return mono
  if (origRate === targetRate) {
    return mono;
  }

  // 2. Linear interpolation resampling
  const ratio = origRate / targetRate;
  const newLength = Math.round(origLength / ratio);
  const resampled = new Float32Array(newLength);

  for (let i = 0; i < newLength; i++) {
    const origIndex = i * ratio;
    const indexFloor = Math.floor(origIndex);
    const indexCeil = Math.min(origLength - 1, Math.ceil(origIndex));
    const fraction = origIndex - indexFloor;
    resampled[i] = mono[indexFloor] * (1 - fraction) + mono[indexCeil] * fraction;
  }

  return resampled;
}
