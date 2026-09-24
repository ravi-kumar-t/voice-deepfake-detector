import React from 'react';
import { Layers, Clock, Activity, AlertCircle, ShieldCheck, AlertTriangle, Info, Sparkles } from 'lucide-react';

export default function SegmentAnalysis({
  audioBuffer,
  result,
  segmentData,
  selectedSegment,
  onSelectSegment,
  isAnalyzingSegments,
  segmentError
}) {
  // Calculate real RMS Acoustic Energy for each 0.5s slice from the audio buffer
  const calculateSegmentAcousticEnergy = () => {
    if (!audioBuffer) return {};

    const sampleRate = audioBuffer.sampleRate;
    const channelData = audioBuffer.getChannelData(0);
    const duration = audioBuffer.duration;
    const segmentDuration = 0.5;
    const totalSegments = Math.min(4, Math.max(1, Math.ceil(duration / segmentDuration)));

    const energyMap = {};
    for (let i = 0; i < totalSegments; i++) {
      const startTime = i * segmentDuration;
      const endTime = Math.min(duration, (i + 1) * segmentDuration);
      const startSample = Math.floor(startTime * sampleRate);
      const endSample = Math.min(channelData.length, Math.floor(endTime * sampleRate));

      let sumSquares = 0;
      const count = endSample - startSample;
      for (let s = startSample; s < endSample; s++) {
        sumSquares += channelData[s] * channelData[s];
      }
      const rms = count > 0 ? Math.sqrt(sumSquares / count) : 0;
      const energyDb = rms > 0 ? 20 * Math.log10(rms) : -80;
      const energyPercent = Math.min(100, Math.max(0, ((energyDb + 60) / 60) * 100));

      energyMap[i] = {
        rms: rms.toFixed(4),
        energyPercent: energyPercent.toFixed(1),
        activity: energyPercent > 15 ? 'Active Phonation' : 'Low Energy / Silence'
      };
    }

    return energyMap;
  };

  const energyMap = calculateSegmentAcousticEnergy();

  // Segments to display (either from real model inference or default slots)
  const defaultSlots = [
    { id: 1, start: 0.0, end: 0.5, timeRange: '0.0 — 0.5s' },
    { id: 2, start: 0.5, end: 1.0, timeRange: '0.5 — 1.0s' },
    { id: 3, start: 1.0, end: 1.5, timeRange: '1.0 — 1.5s' },
    { id: 4, start: 1.5, end: 2.0, timeRange: '1.5 — 2.0s' }
  ];

  const modelSegments = segmentData && segmentData.segments ? segmentData.segments : null;

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <Layers size={18} />
          <span>SEGMENT-LEVEL MODEL EVIDENCE</span>
        </div>
        <span className="panel-badge">BiLSTM Temporal Slices (500ms)</span>
      </div>

      <div className="panel-body">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px', flexWrap: 'wrap', gap: '8px' }}>
          <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
            Model evidence from temporal regions of the full 2-second audio. Click a segment to highlight its spectral region.
          </p>
          {selectedSegment && (
            <button
              className="btn-secondary"
              style={{ fontSize: '0.74rem', padding: '3px 8px' }}
              onClick={() => onSelectSegment(null)}
            >
              Clear Highlight
            </button>
          )}
        </div>

        {segmentError && (
          <div style={{
            marginBottom: '14px',
            padding: '8px 12px',
            borderRadius: '6px',
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            color: '#f43f5e',
            fontSize: '0.8rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <AlertCircle size={14} />
            <span>Segment Analysis Notice: {segmentError}</span>
          </div>
        )}

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '14px' }}>
          {defaultSlots.map((slot, idx) => {
            const seg = modelSegments ? modelSegments[idx] : null;
            const energy = energyMap[idx];
            const isSelected = selectedSegment && selectedSegment.start === slot.start && selectedSegment.end === slot.end;

            const isReal = seg ? seg.prediction === 'REAL' : null;
            const synthProbPercent = seg ? (seg.synthetic_probability * 100).toFixed(1) : null;
            const confidencePercent = seg ? seg.confidence.toFixed(1) : null;

            return (
              <div
                key={slot.id}
                onClick={() => onSelectSegment && onSelectSegment(isSelected ? null : { id: slot.id, start: slot.start, end: slot.end })}
                style={{
                  padding: '16px',
                  borderRadius: '12px',
                  background: isSelected
                    ? 'rgba(56, 189, 248, 0.12)'
                    : 'rgba(255, 255, 255, 0.02)',
                  border: isSelected
                    ? '1.5px solid #38bdf8'
                    : seg
                      ? (isReal ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(244, 63, 94, 0.3)')
                      : '1px solid rgba(255, 255, 255, 0.06)',
                  boxShadow: isSelected ? '0 0 16px rgba(56, 189, 248, 0.25)' : 'none',
                  cursor: 'pointer',
                  display: 'flex',
                  flexDirection: 'column',
                  gap: '10px',
                  transition: 'all 0.2s ease'
                }}
              >
                {/* Header / Time range */}
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                    <Clock size={14} color="#38bdf8" />
                    <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', fontSize: '0.90rem', color: '#f8fafc' }}>
                      {slot.timeRange}
                    </span>
                  </div>

                  {seg ? (
                    <span style={{
                      fontSize: '0.74rem',
                      fontFamily: 'var(--font-mono)',
                      fontWeight: '700',
                      padding: '2px 8px',
                      borderRadius: '4px',
                      background: isReal ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
                      color: isReal ? '#34d399' : '#f87171',
                      border: `1px solid ${isReal ? 'rgba(16, 185, 129, 0.35)' : 'rgba(244, 63, 94, 0.35)'}`
                    }}>
                      {seg.prediction}
                    </span>
                  ) : (
                    <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                      {isAnalyzingSegments ? 'Evaluating...' : 'Awaiting'}
                    </span>
                  )}
                </div>

                {/* Model Prediction Metrics */}
                {seg ? (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', background: 'rgba(0,0,0,0.2)', padding: '8px 10px', borderRadius: '8px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Synthetic probability:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: isReal ? '#34d399' : '#f87171' }}>
                        {synthProbPercent}%
                      </span>
                    </div>

                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem' }}>
                      <span style={{ color: 'var(--text-secondary)' }}>Confidence:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', fontWeight: '600', color: '#f1f5f9' }}>
                        {confidencePercent}%
                      </span>
                    </div>

                    {/* Mini Probability Bar */}
                    <div style={{
                      height: '4px',
                      borderRadius: '2px',
                      background: 'rgba(255, 255, 255, 0.08)',
                      overflow: 'hidden',
                      marginTop: '2px'
                    }}>
                      <div
                        style={{
                          height: '100%',
                          width: `${Math.max(1, Math.min(100, synthProbPercent))}%`,
                          background: isReal ? '#10b981' : '#f43f5e',
                          borderRadius: '2px'
                        }}
                      />
                    </div>
                  </div>
                ) : (
                  <div style={{ fontSize: '0.76rem', color: 'var(--text-muted)', padding: '6px 0' }}>
                    Run analysis to compute segment deepfake score.
                  </div>
                )}

                {/* Acoustic Energy (RMS) - Explicitly separated and labeled */}
                {energy && (
                  <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '6px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.74rem', color: 'var(--text-muted)' }}>
                      <span>Acoustic Energy:</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: '#94a3b8' }}>
                        {energy.energyPercent}% ({energy.activity})
                      </span>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Required Forensics Explanatory Note */}
        <div style={{
          marginTop: '16px',
          padding: '12px 16px',
          borderRadius: '8px',
          background: 'rgba(56, 189, 248, 0.04)',
          border: '1px solid rgba(56, 189, 248, 0.15)',
          fontSize: '0.78rem',
          color: 'var(--text-secondary)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '10px'
        }}>
          <Info size={16} color="#38bdf8" style={{ flexShrink: 0, marginTop: '2px' }} />
          <span>
            Model evidence from temporal regions of the full 2-second audio. Segment scores represent localized temporal representations and should be interpreted as localized evidence, not definitive proof of manipulation.
          </span>
        </div>
      </div>
    </div>
  );
}
