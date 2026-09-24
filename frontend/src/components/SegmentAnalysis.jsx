import React from 'react';
import { Layers, Clock, Activity, AlertCircle, ShieldCheck } from 'lucide-react';

export default function SegmentAnalysis({ audioBuffer, result }) {
  // If audio buffer is available, calculate real RMS energy for each 0.5s slice
  const calculateSegmentMetrics = () => {
    if (!audioBuffer) return [];

    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);
    const duration = audioBuffer.duration;
    const segmentDuration = 0.5;
    const totalSegments = Math.min(4, Math.max(1, Math.ceil(duration / segmentDuration)));

    const segments = [];
    for (let i = 0; i < totalSegments; i++) {
      const startTime = i * segmentDuration;
      const endTime = Math.min(duration, (i + 1) * segmentDuration);
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.min(channelData.length, Math.floor(endTime * sampleRate));

      // Calculate RMS energy of this actual audio slice
      let sumSquares = 0;
      const count = endSample - startSample;
      for (let s = startSample; s < endSample; s++) {
        sumSquares += channelData[s] * channelData[s];
      }
      const rms = count > 0 ? Math.sqrt(sumSquares / count) : 0;
      const energyDb = rms > 0 ? 20 * Math.log10(rms) : -80;
      const energyPercent = Math.min(100, Math.max(0, ((energyDb + 60) / 60) * 100));

      segments.push({
        id: i + 1,
        timeRange: `${startTime.toFixed(1)}–${endTime.toFixed(1)}s`,
        rmsEnergy: rms.toFixed(4),
        energyPercent: energyPercent.toFixed(1),
        status: energyPercent > 15 ? 'Active Phonation' : 'Low Energy / Silence'
      });
    }

    return segments;
  };

  const segments = calculateSegmentMetrics();

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={18} />
          <span>Segment-Level Analysis</span>
        </div>
        <span className="panel-badge">Temporal Windowing (0.5s Slices)</span>
      </div>

      <div className="panel-body">
        <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', marginBottom: '16px' }}>
          Deconstructs the 2.0-second input clip into 500ms time windows to inspect temporal stability and localized acoustic activity.
        </p>

        {segments.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '24px',
            color: 'var(--text-muted)',
            fontSize: '0.84rem'
          }}>
            Awaiting audio file to compute segment windows.
          </div>
        ) : (
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '12px' }}>
            {segments.map((seg) => (
              <div
                key={seg.id}
                style={{
                  padding: '14px',
                  borderRadius: '10px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(56, 189, 248, 0.15)',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '8px'
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} color="#38bdf8" />
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', fontSize: '0.88rem', color: '#f8fafc' }}>
                      {seg.timeRange}
                    </span>
                  </div>
                  <span style={{
                    fontSize: '0.7rem',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    background: 'rgba(56, 189, 248, 0.1)',
                    color: '#38bdf8',
                    fontFamily: 'var(--font-mono)'
                  }}>
                    Seg {seg.id}
                  </span>
                </div>

                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  <span>Acoustic Energy:</span>
                  <span style={{ fontFamily: 'var(--font-mono)', color: '#f1f5f9' }}>{seg.energyPercent}%</span>
                </div>

                {/* Energy mini bar */}
                <div style={{
                  height: '6px',
                  borderRadius: '3px',
                  background: 'rgba(255, 255, 255, 0.08)',
                  overflow: 'hidden'
                }}>
                  <div
                    style={{
                      height: '100%',
                      width: `${seg.energyPercent}%`,
                      background: 'linear-gradient(90deg, #38bdf8, #818cf8)',
                      borderRadius: '3px'
                    }}
                  />
                </div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  fontSize: '0.74rem',
                  marginTop: '2px'
                }}>
                  <span style={{ color: 'var(--text-muted)' }}>Status:</span>
                  <span style={{ color: seg.status.includes('Active') ? '#34d399' : '#94a3b8' }}>
                    {seg.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}

        <div style={{
          marginTop: '16px',
          padding: '10px 12px',
          borderRadius: '8px',
          background: 'rgba(56, 189, 248, 0.04)',
          border: '1px solid rgba(56, 189, 248, 0.12)',
          fontSize: '0.76rem',
          color: 'var(--text-muted)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <Activity size={14} color="#38bdf8" />
          <span>BiLSTM analyzes sequential acoustic transitions across all 4 consecutive segment windows simultaneously.</span>
        </div>
      </div>
    </div>
  );
}
