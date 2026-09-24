import React, { useRef, useEffect } from 'react';
import { Waves, Zap, Info } from 'lucide-react';
import { computeSpectrogram, getSpectrogramColor } from '../utils/audioHelper';

export default function SpectrogramViewer({
  audioBuffer,
  currentTime,
  duration,
  selectedSegment
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Background
    ctx.fillStyle = '#080c18';
    ctx.fillRect(0, 0, width, height);

    if (!audioBuffer) {
      // Idle state
      ctx.fillStyle = 'rgba(100, 116, 139, 0.4)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('Awaiting audio file for spectral decomposition', width / 2, height / 2);
      return;
    }

    // Compute real STFT spectrogram
    const spec = computeSpectrogram(audioBuffer, 512, 128);
    const { frames, numBins, numFrames } = spec;

    if (!frames || frames.length === 0) return;

    // Draw spectrogram heatmap
    const cellWidth = width / numFrames;
    const cellHeight = height / numBins;

    // Offscreen / Direct canvas pixel painting
    const imgData = ctx.createImageData(width, height);
    const data = imgData.data;

    // Map time frames (X) and frequency bins (Y, inverted so 0Hz is at bottom)
    for (let x = 0; x < width; x++) {
      const frameIdx = Math.min(numFrames - 1, Math.floor((x / width) * numFrames));
      const frame = frames[frameIdx];

      for (let y = 0; y < height; y++) {
        // Invert Y: top is high frequency (8 kHz), bottom is low frequency (0 Hz)
        const binIdx = Math.min(numBins - 1, Math.floor(((height - 1 - y) / height) * numBins));
        const val = frame[binIdx]; // normalized 0.0 to 1.0

        // Parse color components
        const rgb = getRGB(val);
        const pixelIdx = (y * width + x) * 4;
        data[pixelIdx] = rgb[0];     // R
        data[pixelIdx + 1] = rgb[1]; // G
        data[pixelIdx + 2] = rgb[2]; // B
        data[pixelIdx + 3] = 255;    // A
      }
    }

    ctx.putImageData(imgData, 0, 0);

    // Draw Selected Segment Highlight Overlay
    if (selectedSegment && duration > 0) {
      const segStartX = (selectedSegment.start / duration) * width;
      const segEndX = (Math.min(selectedSegment.end, duration) / duration) * width;
      const segWidth = Math.max(2, segEndX - segStartX);

      ctx.fillStyle = 'rgba(56, 189, 248, 0.22)';
      ctx.fillRect(segStartX, 0, segWidth, height);

      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.strokeRect(segStartX, 0, segWidth, height);

      // Label on top
      ctx.fillStyle = '#38bdf8';
      ctx.font = '10px "JetBrains Mono", monospace';
      ctx.fillText(`Seg [${selectedSegment.start.toFixed(1)}–${selectedSegment.end.toFixed(1)}s]`, segStartX + 4, 14);
    }

    // Draw Grid & Frequency Guide Lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
    ctx.lineWidth = 1;

    // Frequency markers (e.g. 2kHz, 4kHz, 6kHz, 8kHz)
    const freqSteps = [
      { label: '8 kHz', y: 14 },
      { label: '6 kHz', y: height * 0.25 },
      { label: '4 kHz', y: height * 0.5 },
      { label: '2 kHz', y: height * 0.75 },
      { label: '0 Hz', y: height - 6 }
    ];

    ctx.fillStyle = 'rgba(255, 255, 255, 0.75)';
    ctx.font = '9px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';

    freqSteps.forEach(f => {
      ctx.beginPath();
      ctx.moveTo(0, f.y);
      ctx.lineTo(width, f.y);
      ctx.stroke();
      ctx.fillText(f.label, 8, f.y > 20 ? f.y - 3 : f.y + 10);
    });

    // Draw Playhead cursor
    if (duration > 0 && currentTime >= 0) {
      const playheadX = (currentTime / duration) * width;
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 2;
      ctx.shadowColor = '#38bdf8';
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();
      ctx.shadowBlur = 0; // reset
    }

  }, [audioBuffer, currentTime, duration, selectedSegment]);

  // Fast RGB lookup for heatmap
  const getRGB = (val) => {
    if (val < 0.2) {
      const t = val / 0.2;
      return [Math.round(8 + t * 40), Math.round(12 + t * 15), Math.round(30 + t * 90)];
    } else if (val < 0.45) {
      const t = (val - 0.2) / 0.25;
      return [Math.round(48 + t * 140), Math.round(27 + t * 12), Math.round(120 + t * 25)];
    } else if (val < 0.7) {
      const t = (val - 0.45) / 0.25;
      return [Math.round(188 + t * 60), Math.round(39 + t * 76), Math.round(145 - t * 120)];
    } else if (val < 0.9) {
      const t = (val - 0.7) / 0.2;
      return [Math.round(248 + t * 7), Math.round(115 + t * 100), Math.round(25 + t * 100)];
    } else {
      const t = (val - 0.9) / 0.1;
      return [255, Math.round(215 + t * 40), Math.round(125 + t * 130)];
    }
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Waves size={18} />
          <span>Spectral Analysis</span>
        </div>
        <span className="panel-badge">Audio Spectral Representation</span>
      </div>

      <div className="panel-body" style={{ padding: '16px' }}>
        <div style={{ position: 'relative', width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
          <canvas
            ref={canvasRef}
            width={720}
            height={200}
            style={{
              width: '100%',
              height: '200px',
              display: 'block'
            }}
          />
        </div>

        {/* Legend & Axes Labeling */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '12px',
          flexWrap: 'wrap',
          gap: '10px',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontFamily: 'var(--font-mono)' }}>Time (0 → {duration ? `${duration.toFixed(1)}s` : '2.0s'})</span>
            <span>•</span>
            <span style={{ fontFamily: 'var(--font-mono)' }}>Frequency (0 → 8.0 kHz)</span>
          </div>

          {/* Color scale gradient bar */}
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>-80 dB</span>
            <div style={{
              width: '100px',
              height: '8px',
              borderRadius: '4px',
              background: 'linear-gradient(90deg, #080c1e 0%, #4c1d95 25%, #db2777 50%, #f97316 75%, #ffffff 100%)',
              border: '1px solid rgba(255, 255, 255, 0.15)'
            }} />
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>0 dB</span>
          </div>
        </div>

        <div style={{
          marginTop: '8px',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          fontSize: '0.74rem',
          color: 'var(--text-muted)'
        }}>
          <Info size={13} color="#38bdf8" />
          <span>Real-time STFT FFT computed across audio buffer. Visualizes formant distribution and harmonic spectral energy.</span>
        </div>
      </div>
    </div>
  );
}
