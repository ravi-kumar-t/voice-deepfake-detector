import React from 'react';
import { ShieldCheck, AlertTriangle, Activity, CheckCircle, Fingerprint, Award } from 'lucide-react';

export default function DetectionResult({ result }) {
  if (!result) {
    return (
      <div className="glass-panel" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
        <div className="panel-header">
          <div className="panel-title">
            <Activity size={18} />
            <span>Detection Result</span>
          </div>
          <span className="panel-badge">Awaiting Inference</span>
        </div>
        <div className="panel-body" style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: '220px',
          color: 'var(--text-muted)',
          textAlign: 'center',
          gap: '12px'
        }}>
          <Fingerprint size={48} strokeWidth={1.5} color="rgba(56, 189, 248, 0.3)" />
          <div>
            <p style={{ fontWeight: '500', color: 'var(--text-secondary)' }}>No Audio Analyzed Yet</p>
            <p style={{ fontSize: '0.82rem', marginTop: '4px' }}>
              Upload or record a WAV sample and click "Analyze Audio" to generate forensic results.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const isReal = result.prediction === 'REAL';
  const synthProbPercent = (result.synthetic_probability * 100).toFixed(2);
  const confidencePercent = result.confidence.toFixed(2);

  return (
    <div
      className="glass-panel"
      style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100%',
        borderColor: isReal ? 'var(--real-border)' : 'var(--fake-border)',
        boxShadow: isReal ? 'var(--shadow-glow-real)' : 'var(--shadow-glow-fake)',
        transition: 'all 0.3s cubic-bezier(0.16, 1, 0.3, 1)'
      }}
    >
      <div className="panel-header" style={{
        background: isReal ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)'
      }}>
        <div className="panel-title">
          {isReal ? <ShieldCheck size={20} color="var(--real-emerald)" /> : <AlertTriangle size={20} color="var(--fake-crimson)" />}
          <span>DETECTION RESULT</span>
        </div>
        <span
          className="panel-badge"
          style={{
            background: isReal ? 'rgba(16, 185, 129, 0.15)' : 'rgba(244, 63, 94, 0.15)',
            borderColor: isReal ? 'rgba(16, 185, 129, 0.4)' : 'rgba(244, 63, 94, 0.4)',
            color: isReal ? '#34d399' : '#f87171'
          }}
        >
          {isReal ? 'VERIFIED AUTHENTIC' : 'SYNTHETIC DETECTED'}
        </span>
      </div>

      <div className="panel-body" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        {/* Large Verdict Hero */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '20px 24px',
          borderRadius: '12px',
          background: isReal ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
          border: `1px solid ${isReal ? 'rgba(16, 185, 129, 0.25)' : 'rgba(244, 63, 94, 0.25)'}`,
          flexWrap: 'wrap',
          gap: '16px'
        }}>
          <div>
            <span style={{
              fontSize: '0.75rem',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              display: 'block',
              marginBottom: '4px'
            }}>
              Classification Verdict
            </span>
            <div style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '2.4rem',
              fontWeight: '800',
              letterSpacing: '0.03em',
              color: isReal ? '#10b981' : '#f43f5e',
              lineHeight: 1
            }}>
              {result.prediction}
            </div>
            <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)', marginTop: '6px' }}>
              File: <code style={{ color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>{result.filename}</code>
            </p>
          </div>

          <div style={{ textAlign: 'right' }}>
            <span style={{
              fontSize: '0.75rem',
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              display: 'block',
              marginBottom: '4px'
            }}>
              Model Confidence
            </span>
            <div style={{
              fontFamily: 'var(--font-heading)',
              fontSize: '2.2rem',
              fontWeight: '700',
              color: '#ffffff',
              lineHeight: 1
            }}>
              {confidencePercent}%
            </div>
            <span style={{ fontSize: '0.78rem', color: isReal ? '#34d399' : '#f87171' }}>
              {isReal ? 'High authenticity match' : 'High synthetic probability'}
            </span>
          </div>
        </div>

        {/* Synthetic Probability Gauge Bar */}
        <div>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px', fontSize: '0.82rem' }}>
            <span style={{ color: 'var(--text-secondary)' }}>Synthetic Probability:</span>
            <span style={{
              fontFamily: 'var(--font-mono)',
              fontWeight: '600',
              color: isReal ? '#34d399' : '#f87171'
            }}>
              {synthProbPercent}% ({result.synthetic_probability})
            </span>
          </div>

          <div style={{
            position: 'relative',
            height: '14px',
            borderRadius: '7px',
            background: 'rgba(255, 255, 255, 0.06)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            overflow: 'hidden'
          }}>
            {/* 50% Threshold line marker */}
            <div style={{
              position: 'absolute',
              left: '50%',
              top: 0,
              bottom: 0,
              width: '2px',
              background: 'rgba(255, 255, 255, 0.4)',
              zIndex: 3
            }} title="Classification Threshold (0.50)" />

            <div
              style={{
                height: '100%',
                width: `${Math.max(1, Math.min(100, synthProbPercent))}%`,
                background: isReal
                  ? 'linear-gradient(90deg, #059669 0%, #10b981 100%)'
                  : 'linear-gradient(90deg, #f59e0b 0%, #f43f5e 100%)',
                borderRadius: '7px',
                transition: 'width 0.6s cubic-bezier(0.16, 1, 0.3, 1)'
              }}
            />
          </div>

          <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '6px', fontSize: '0.7rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
            <span>0.0 (Authentic Real)</span>
            <span>Threshold 0.5</span>
            <span>1.0 (Synthetic Deepfake)</span>
          </div>
        </div>

        {/* Forensic Insights List */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '12px',
          paddingTop: '12px',
          borderTop: '1px solid rgba(255, 255, 255, 0.06)'
        }}>
          <div style={{ padding: '10px 12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '2px' }}>Temporal Consistency</span>
            <span style={{ fontSize: '0.85rem', color: '#f1f5f9', fontWeight: '500' }}>
              {isReal ? 'Natural Human Phonation' : 'Algorithmic Vocoder Markers'}
            </span>
          </div>

          <div style={{ padding: '10px 12px', background: 'rgba(255, 255, 255, 0.02)', borderRadius: '8px', border: '1px solid rgba(255, 255, 255, 0.05)' }}>
            <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)', display: 'block', marginBottom: '2px' }}>Mel-Band Coherence</span>
            <span style={{ fontSize: '0.85rem', color: '#f1f5f9', fontWeight: '500' }}>
              {isReal ? 'Organic Formant Dispersion' : 'High-Frequency Synthesis Artifacts'}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
