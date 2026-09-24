import React from 'react';
import { Shield, Activity, Radio, Cpu, Sparkles } from 'lucide-react';

export default function Header({ backendStatus }) {
  return (
    <header className="glass-panel" style={{ padding: '20px 24px' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '16px' }}>
        {/* Brand & Subtitle */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div style={{
            width: '48px',
            height: '48px',
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(56, 189, 248, 0.2) 0%, rgba(99, 102, 241, 0.2) 100%)',
            border: '1px solid rgba(56, 189, 248, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 20px rgba(56, 189, 248, 0.25)'
          }}>
            <Shield size={26} color="#38bdf8" />
          </div>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <h1 style={{
                fontFamily: 'var(--font-heading)',
                fontSize: '1.65rem',
                fontWeight: '700',
                letterSpacing: '-0.02em',
                background: 'linear-gradient(135deg, #ffffff 30%, #38bdf8 100%)',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                lineHeight: 1.2
              }}>
                VoiceGuard AI
              </h1>
              <span style={{
                fontSize: '0.68rem',
                fontFamily: 'var(--font-mono)',
                padding: '2px 8px',
                borderRadius: '6px',
                background: 'rgba(56, 189, 248, 0.12)',
                color: '#38bdf8',
                border: '1px solid rgba(56, 189, 248, 0.3)',
                fontWeight: 600
              }}>
                v1.0 FORENSICS
              </span>
            </div>
            <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginTop: '2px', fontWeight: '400' }}>
              Generative Voice & Audio Deepfake Detection Suite
            </p>
          </div>
        </div>

        {/* Status Indicators & Metadata */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '14px', flexWrap: 'wrap' }}>
          {/* Engine Spec Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '7px 12px',
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.03)',
            border: '1px solid rgba(255, 255, 255, 0.08)',
            fontSize: '0.78rem',
            color: 'var(--text-secondary)'
          }}>
            <Cpu size={14} color="#818cf8" />
            <span>CNN-BiLSTM Pipeline</span>
          </div>

          {/* Backend Status Badge */}
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            padding: '7px 14px',
            borderRadius: '8px',
            background: backendStatus.online ? 'rgba(16, 185, 129, 0.08)' : 'rgba(244, 63, 94, 0.08)',
            border: `1px solid ${backendStatus.online ? 'rgba(16, 185, 129, 0.3)' : 'rgba(244, 63, 94, 0.3)'}`,
            fontSize: '0.82rem',
            fontWeight: '500'
          }}>
            <span className={`pulse-dot ${backendStatus.online ? 'online' : 'offline'}`} />
            <span style={{ color: backendStatus.online ? '#34d399' : '#f87171' }}>
              {backendStatus.online ? 'Backend Online' : 'Backend Offline'}
            </span>
          </div>
        </div>
      </div>

      <div style={{
        marginTop: '14px',
        paddingTop: '12px',
        borderTop: '1px solid rgba(255, 255, 255, 0.05)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '0.83rem',
        color: 'var(--text-muted)'
      }}>
        <span>Analyze voice recordings for synthetic or manipulated audio using spectral and temporal deepfake analysis.</span>
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.75rem', color: '#64748b' }}>
          Model: 80-Mel PyTorch / BiLSTM
        </span>
      </div>
    </header>
  );
}
