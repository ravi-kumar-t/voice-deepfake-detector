import React, { useRef, useEffect } from 'react';
import { Play, Pause, Volume2, VolumeX, RotateCcw } from 'lucide-react';

export default function AudioPlayer({
  audioUrl,
  isPlaying,
  currentTime,
  duration,
  volume,
  isMuted,
  onPlayToggle,
  onSeek,
  onVolumeChange,
  onMuteToggle,
  onRestart
}) {
  const formatTime = (secs) => {
    if (isNaN(secs) || secs < 0) return '00:00.00';
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 100);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`;
  };

  const progressPercent = duration > 0 ? Math.min(100, (currentTime / duration) * 100) : 0;

  return (
    <div className="glass-panel" style={{ padding: '16px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '16px', flexWrap: 'wrap' }}>
        {/* Play / Pause / Restart */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={onPlayToggle}
            disabled={!audioUrl}
            style={{
              width: '42px',
              height: '42px',
              borderRadius: '50%',
              background: 'linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%)',
              border: '1px solid rgba(56, 189, 248, 0.4)',
              color: '#ffffff',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              cursor: audioUrl ? 'pointer' : 'not-allowed',
              opacity: audioUrl ? 1 : 0.5,
              boxShadow: '0 0 15px rgba(56, 189, 248, 0.3)',
              transition: 'all 0.2s ease'
            }}
            title={isPlaying ? 'Pause' : 'Play'}
          >
            {isPlaying ? <Pause size={18} /> : <Play size={18} style={{ marginLeft: '2px' }} />}
          </button>

          <button
            className="btn-secondary"
            onClick={onRestart}
            disabled={!audioUrl}
            style={{ padding: '8px 10px', borderRadius: '50%' }}
            title="Restart from beginning"
          >
            <RotateCcw size={14} />
          </button>
        </div>

        {/* Time Progress Bar */}
        <div style={{ flex: 1, minWidth: '200px', display: 'flex', flexDirection: 'column', gap: '6px' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.78rem', fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
            <span>{formatTime(currentTime)}</span>
            <span>{formatTime(duration)}</span>
          </div>

          <div
            onClick={(e) => {
              if (!audioUrl || !duration) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const clickPos = (e.clientX - rect.left) / rect.width;
              onSeek(clickPos * duration);
            }}
            style={{
              height: '8px',
              borderRadius: '4px',
              background: 'rgba(255, 255, 255, 0.08)',
              position: 'relative',
              cursor: audioUrl ? 'pointer' : 'default',
              overflow: 'hidden'
            }}
          >
            <div
              style={{
                position: 'absolute',
                left: 0,
                top: 0,
                bottom: 0,
                width: `${progressPercent}%`,
                background: 'linear-gradient(90deg, #38bdf8 0%, #818cf8 100%)',
                borderRadius: '4px',
                transition: 'width 0.05s linear'
              }}
            />
          </div>
        </div>

        {/* Volume Controls */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            className="btn-secondary"
            onClick={onMuteToggle}
            style={{ padding: '6px 8px', borderRadius: '6px' }}
            title={isMuted ? 'Unmute' : 'Mute'}
          >
            {isMuted || volume === 0 ? <VolumeX size={15} color="#f43f5e" /> : <Volume2 size={15} color="var(--accent-cyan)" />}
          </button>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={isMuted ? 0 : volume}
            onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
            style={{
              width: '70px',
              accentColor: 'var(--accent-cyan)',
              cursor: 'pointer'
            }}
          />
        </div>
      </div>
    </div>
  );
}
