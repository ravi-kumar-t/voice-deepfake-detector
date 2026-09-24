import React from 'react';
import { HelpCircle, Waves, Cpu, Clock, CheckCircle } from 'lucide-react';

export default function HowItWorks() {
  const steps = [
    {
      num: '01',
      title: 'Spectral Analysis',
      desc: 'Audio is converted into an 80-bin Mel-spectrogram representation capturing acoustic frequencies over time.',
      icon: Waves,
      color: '#38bdf8'
    },
    {
      num: '02',
      title: 'Feature Extraction',
      desc: '2D Convolutional layers learn spatial and harmonic spectral patterns associated with synthetic vocoders.',
      icon: Cpu,
      color: '#818cf8'
    },
    {
      num: '03',
      title: 'Temporal Analysis',
      desc: 'Bidirectional LSTM layers analyze how phoneme transitions and spectral features evolve over time.',
      icon: Clock,
      color: '#a855f7'
    }
  ];

  return (
    <div className="glass-panel">
      <div className="panel-header">
        <div className="panel-title">
          <HelpCircle size={18} />
          <span>How the Detection Works</span>
        </div>
        <span className="panel-badge">Forensic Methodology</span>
      </div>

      <div className="panel-body">
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: '16px' }}>
          {steps.map((step, idx) => {
            const Icon = step.icon;
            return (
              <div
                key={idx}
                style={{
                  padding: '16px',
                  borderRadius: '12px',
                  background: 'rgba(255, 255, 255, 0.02)',
                  border: '1px solid rgba(255, 255, 255, 0.06)',
                  position: 'relative',
                  overflow: 'hidden'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
                  <span style={{
                    fontFamily: 'var(--font-mono)',
                    fontSize: '1.4rem',
                    fontWeight: '800',
                    color: step.color,
                    lineHeight: 1
                  }}>
                    {step.num}
                  </span>
                  <div style={{
                    width: '32px',
                    height: '32px',
                    borderRadius: '8px',
                    background: `${step.color}15`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: step.color
                  }}>
                    <Icon size={16} />
                  </div>
                </div>

                <h3 style={{
                  fontFamily: 'var(--font-heading)',
                  fontSize: '1rem',
                  fontWeight: '600',
                  color: '#f8fafc',
                  marginBottom: '6px'
                }}>
                  {step.title}
                </h3>

                <p style={{
                  fontSize: '0.82rem',
                  color: 'var(--text-secondary)',
                  lineHeight: '1.45'
                }}>
                  {step.desc}
                </p>
              </div>
            );
          })}
        </div>

        {/* Final Classification summary */}
        <div style={{
          marginTop: '16px',
          padding: '14px 18px',
          borderRadius: '10px',
          background: 'linear-gradient(90deg, rgba(56, 189, 248, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%)',
          border: '1px solid rgba(56, 189, 248, 0.2)',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <CheckCircle size={20} color="#38bdf8" style={{ flexShrink: 0 }} />
          <div>
            <span style={{ fontWeight: '600', fontSize: '0.88rem', color: '#f8fafc' }}>
              Final Decision:
            </span>
            <span style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', marginLeft: '6px' }}>
              The model computes temporal average pooling and evaluates binary logit probability to classify audio as <strong>REAL (Authentic Human)</strong> or <strong>SYNTHETIC (AI Generated)</strong>.
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
