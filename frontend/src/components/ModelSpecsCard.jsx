import React from 'react';
import { Cpu, CheckCircle2, Shield, Zap, Hash } from 'lucide-react';

export default function ModelSpecsCard() {
  const specs = [
    { label: 'Model Pipeline', value: 'CNN + BiLSTM' },
    { label: 'Audio Input', value: '16 kHz Mono WAV' },
    { label: 'Analysis Window', value: '2.0 seconds (32,000 samples)' },
    { label: 'Spectral Features', value: '80-bin Mel Spectrogram' },
    { label: 'Frequency Range', value: '0 – 8,000 Hz' },
    { label: 'FFT Window / Hop', value: 'N_FFT: 1024, Hop: 256' },
    { label: 'Architecture', value: '2D CNN → BiLSTM → Binary Classifier' },
    { label: 'Inference Latency', value: '~10–15 ms (CPU Optimized)' },
  ];

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Cpu size={18} />
          <span>Model Architecture & Pipeline Specs</span>
        </div>
        <span className="panel-badge">PyTorch Production Checkpoint</span>
      </div>

      <div className="panel-body">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '12px' }}>
          {specs.map((item, idx) => (
            <div
              key={idx}
              style={{
                padding: '10px 14px',
                borderRadius: '8px',
                background: 'rgba(255, 255, 255, 0.02)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                display: 'flex',
                flexDirection: 'column',
                gap: '2px'
              }}
            >
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                {item.label}
              </span>
              <span style={{ fontSize: '0.86rem', color: '#f1f5f9', fontWeight: '500' }}>
                {item.value}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
