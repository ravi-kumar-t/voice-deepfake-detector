/**
 * API Service for Voice Deepfake Detection Backend
 */

export const API_BASE_URL =
  import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

/**
 * Check backend health status
 * @returns {Promise<{ status: string, model_loaded: boolean }>}
 */
export async function checkBackendHealth() {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 3500);

  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Accept': 'application/json' },
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    return {
      online: true,
      status: data.status,
      modelLoaded: Boolean(data.model_loaded),
      error: null
    };
  } catch (err) {
    clearTimeout(timeoutId);
    return {
      online: false,
      status: 'offline',
      modelLoaded: false,
      error: err.name === 'AbortError' ? 'Health check timed out' : err.message
    };
  }
}

/**
 * Send audio file (.wav) to FastAPI /predict endpoint
 * @param {File | Blob} audioFile
 * @param {string} filename
 * @returns {Promise<{ filename: string, prediction: 'REAL' | 'SYNTHETIC', synthetic_probability: number, confidence: number }>}
 */
export async function predictAudio(audioFile, filename = 'recording.wav') {
  const formData = new FormData();
  
  // Ensure the file object has the proper name
  if (audioFile instanceof Blob && !(audioFile instanceof File)) {
    formData.append('file', audioFile, filename);
  } else {
    formData.append('file', audioFile);
  }

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Server returned HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  const data = await response.json();
  return data;
}

/**
 * Send audio file (.wav) to FastAPI /predict-segments endpoint
 * @param {File | Blob} audioFile
 * @param {string} filename
 * @returns {Promise<{ filename: string, segments: Array<{ start: number, end: number, prediction: 'REAL' | 'SYNTHETIC', synthetic_probability: number, confidence: number }> }>}
 */
export async function predictSegments(audioFile, filename = 'recording.wav') {
  const formData = new FormData();
  
  if (audioFile instanceof Blob && !(audioFile instanceof File)) {
    formData.append('file', audioFile, filename);
  } else {
    formData.append('file', audioFile);
  }

  const response = await fetch(`${API_BASE_URL}/predict-segments`, {
    method: 'POST',
    body: formData,
  });

  if (!response.ok) {
    let errorDetail = `Server returned HTTP ${response.status}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) {
        errorDetail = errJson.detail;
      }
    } catch {
      // fallback
    }
    throw new Error(errorDetail);
  }

  const data = await response.json();
  return data;
}
