import React, { useRef, useEffect } from 'react';
import { Activity, Radio } from 'lucide-react';
import { extractWaveformPeaks } from '../utils/audioHelper';

export default function WaveformViewer({
  audioBuffer,
  currentTime,
  duration,
  onSeek
}) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    // Clear background
    ctx.fillStyle = '#0a0f1d';
    ctx.fillRect(0, 0, width, height);

    // Draw subtle grid lines
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)';
    ctx.lineWidth = 1;
    for (let y = 20; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Zero-crossing center line
    const centerY = height / 2;
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.2)';
    ctx.beginPath();
    ctx.moveTo(0, centerY);
    ctx.lineTo(width, centerY);
    ctx.stroke();

    if (!audioBuffer) {
      // Draw idle placeholder wave
      ctx.fillStyle = 'rgba(100, 116, 139, 0.4)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('No audio waveform loaded', width / 2, centerY + 4);
      return;
    }

    // Extract actual peaks
    const peaks = extractWaveformPeaks(audioBuffer, width);
    const numPeaks = peaks.length;
    const barWidth = width / numPeaks;

    // Draw Waveform Bars
    for (let i = 0; i < numPeaks; i++) {
      const peak = peaks[i];
      const x = i * barWidth;
      const topY = centerY - peak.max * (centerY - 8);
      const bottomY = centerY - peak.min * (centerY - 8);
      const h = Math.max(2, bottomY - topY);

      // Gradient based on amplitude
      const gradient = ctx.createLinearGradient(0, topY, 0, bottomY);
      gradient.addColorStop(0, '#38bdf8');
      gradient.addColorStop(0.5, '#6366f1');
      gradient.addColorStop(1, '#38bdf8');

      ctx.fillStyle = gradient;
      ctx.fillRect(x, topY, Math.max(1, barWidth - 0.5), h);
    }

    // Draw Playhead cursor
    if (duration > 0 && currentTime >= 0) {
      const playheadX = (currentTime / duration) * width;
      
      // Glow line
      ctx.strokeStyle = '#f43f5e';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(playheadX, 0);
      ctx.lineTo(playheadX, height);
      ctx.stroke();

      // Playhead head marker
      ctx.fillStyle = '#f43f5e';
      ctx.beginPath();
      ctx.arc(playheadX, 6, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    // Time Axis Scale Markers
    ctx.fillStyle = 'rgba(148, 163, 184, 0.6)';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.textAlign = 'left';

    const numTicks = 5;
    for (let t = 0; t <= numTicks; t++) {
      const timeVal = (t / numTicks) * (duration || 2.0);
      const tickX = (t / numTicks) * (width - 30);
      ctx.fillText(`${timeVal.toFixed(1)}s`, tickX + 4, height - 6);
    }

  }, [audioBuffer, currentTime, duration]);

  const handleCanvasClick = (e) => {
    if (!audioBuffer || !duration || !onSeek) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const seekTime = (clickX / rect.width) * duration;
    onSeek(Math.max(0, Math.min(duration, seekTime)));
  };

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Activity size={18} />
          <span>Amplitude Waveform Visualizer</span>
        </div>
        <span className="panel-badge">PCM Amplitude / Time</span>
      </div>

      <div className="panel-body" style={{ padding: '16px' }}>
        <div style={{ position: 'relative', width: '100%', borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(56, 189, 248, 0.2)' }}>
          <canvas
            ref={canvasRef}
            width={720}
            height={160}
            onClick={handleCanvasClick}
            style={{
              width: '100%',
              height: '160px',
              display: 'block',
              cursor: audioBuffer ? 'pointer' : 'default'
            }}
          />
        </div>

        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginTop: '10px',
          fontSize: '0.76rem',
          color: 'var(--text-muted)',
          fontFamily: 'var(--font-mono)'
        }}>
          <span>Y-Axis: Normalized Peak Amplitude (-1.0 to +1.0)</span>
          <span>Click waveform to seek</span>
        </div>
      </div>
    </div>
  );
}
