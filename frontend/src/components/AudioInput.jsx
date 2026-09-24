import React, { useState, useRef } from 'react';
import { UploadCloud, Mic, Square, Trash2, FileAudio, AlertCircle, CheckCircle2, Volume2, Sparkles, PlayCircle } from 'lucide-react';
import { encodeWAV } from '../utils/wavEncoder';

export default function AudioInput({
  audioFile,
  audioMetadata,
  onSelectFile,
  onClearFile,
  onAnalyze,
  isAnalyzing,
  analysisStage,
  error,
  onError
}) {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'record'
  const [isDragging, setIsDragging] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);

  const fileInputRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const audioContextRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const pcmChunksRef = useRef([]);
  const timerIntervalRef = useRef(null);

  // --------------------------------------------------------------------------
  // File Upload Handlers
  // --------------------------------------------------------------------------
  const handleFile = (file) => {
    if (!file) return;

    const ext = file.name.split('.').pop().toLowerCase();
    if (ext !== 'wav' && file.type !== 'audio/wav' && file.type !== 'audio/x-wav') {
      onError(`Unsupported file format (.${ext}). VoiceGuard AI currently accepts WAV audio files.`);
      return;
    }

    if (file.size === 0) {
      onError('The selected audio file is empty (0 bytes).');
      return;
    }

    onError(null);
    onSelectFile(file, file.name);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  // --------------------------------------------------------------------------
  // Demo Sample Loader (Fast Hackathon Testing)
  // --------------------------------------------------------------------------
  const loadDemoSample = async (filename) => {
    try {
      onError(null);
      const response = await fetch(`/samples/${filename}`);
      if (!response.ok) {
        throw new Error(`Failed to load demo sample: ${response.statusText}`);
      }
      const blob = await response.blob();
      const file = new File([blob], filename, { type: 'audio/wav' });
      onSelectFile(file, filename);
    } catch (err) {
      onError(`Could not load demo sample: ${err.message}`);
    }
  };

  // --------------------------------------------------------------------------
  // Microphone Recording Handlers (using Web Audio API + 16-bit PCM Encoder)
  // --------------------------------------------------------------------------
  const startRecording = async () => {
    try {
      onError(null);
      pcmChunksRef.current = [];
      setRecordingSeconds(0);

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: false
        }
      });
      mediaStreamRef.current = stream;

      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const audioCtx = new AudioContextClass({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const sourceNode = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        pcmChunksRef.current.push(new Float32Array(inputData));
      };

      sourceNode.connect(processor);
      processor.connect(audioCtx.destination);

      setIsRecording(true);

      const startTime = Date.now();
      timerIntervalRef.current = setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        setRecordingSeconds(elapsed);
        if (elapsed >= 5.0) {
          stopRecording();
        }
      }, 100);

    } catch (err) {
      onError(`Microphone access error: ${err.message || 'Permission denied'}`);
      setIsRecording(false);
    }
  };

  const stopRecording = async () => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current);
      timerIntervalRef.current = null;
    }

    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    setIsRecording(false);

    const chunks = pcmChunksRef.current;
    if (!chunks || chunks.length === 0) {
      onError('No audio was captured from microphone.');
      return;
    }

    let totalLength = 0;
    for (const chunk of chunks) totalLength += chunk.length;

    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }

    const offlineCtx = new (window.OfflineAudioContext || window.webkitOfflineAudioContext)(1, totalLength, 16000);
    const audioBuffer = offlineCtx.createBuffer(1, totalLength, 16000);
    audioBuffer.copyToChannel(merged, 0);

    const wavBlob = encodeWAV(audioBuffer, 16000);
    const filename = `mic_recording_${Date.now().toString().slice(-4)}.wav`;
    const wavFile = new File([wavBlob], filename, { type: 'audio/wav' });

    onSelectFile(wavFile, filename);
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 KB';
    const kb = bytes / 1024;
    if (kb < 1024) return `${kb.toFixed(1)} KB`;
    return `${(kb / 1024).toFixed(2)} MB`;
  };

  return (
    <div className="glass-panel">
      {/* Panel Header with Mode Tabs */}
      <div className="panel-header">
        <div className="panel-title">
          <FileAudio size={18} />
          <span>Audio Input</span>
        </div>

        <div style={{ display: 'flex', gap: '6px', background: 'rgba(0,0,0,0.3)', padding: '3px', borderRadius: '8px' }}>
          <button
            id="tab-upload"
            className="btn-secondary"
            style={{
              padding: '4px 12px',
              fontSize: '0.78rem',
              background: activeTab === 'upload' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              borderColor: activeTab === 'upload' ? 'rgba(56, 189, 248, 0.4)' : 'transparent',
              color: activeTab === 'upload' ? '#38bdf8' : 'var(--text-secondary)'
            }}
            onClick={() => setActiveTab('upload')}
          >
            <UploadCloud size={13} style={{ marginRight: '4px' }} />
            Upload File
          </button>
          <button
            id="tab-record"
            className="btn-secondary"
            style={{
              padding: '4px 12px',
              fontSize: '0.78rem',
              background: activeTab === 'record' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              borderColor: activeTab === 'record' ? 'rgba(56, 189, 248, 0.4)' : 'transparent',
              color: activeTab === 'record' ? '#38bdf8' : 'var(--text-secondary)'
            }}
            onClick={() => setActiveTab('record')}
          >
            <Mic size={13} style={{ marginRight: '4px' }} />
            Record Mic
          </button>
        </div>
      </div>

      <div className="panel-body">
        {/* Error message */}
        {error && (
          <div style={{
            marginBottom: '16px',
            padding: '10px 14px',
            borderRadius: '8px',
            background: 'rgba(244, 63, 94, 0.1)',
            border: '1px solid rgba(244, 63, 94, 0.3)',
            color: '#f43f5e',
            fontSize: '0.84rem',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}>
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Tab 1: Upload Zone */}
        {activeTab === 'upload' && !audioFile && (
          <div>
            <div
              id="upload-dropzone"
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? 'var(--accent-cyan)' : 'rgba(56, 189, 248, 0.25)'}`,
                borderRadius: '12px',
                padding: '30px 20px',
                textAlign: 'center',
                cursor: 'pointer',
                background: isDragging ? 'rgba(56, 189, 248, 0.08)' : 'rgba(15, 23, 42, 0.4)',
                transition: 'all 0.2s ease',
              }}
            >
              <input
                id="audio-file-input"
                ref={fileInputRef}
                type="file"
                accept=".wav,audio/wav,audio/x-wav"
                style={{ display: 'none' }}
                onChange={(e) => {
                  if (e.target.files?.[0]) handleFile(e.target.files[0]);
                }}
              />
              <div style={{
                width: '50px',
                height: '50px',
                borderRadius: '50%',
                background: 'rgba(56, 189, 248, 0.1)',
                border: '1px solid rgba(56, 189, 248, 0.2)',
                margin: '0 auto 12px auto',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-cyan)'
              }}>
                <UploadCloud size={26} />
              </div>
              <p style={{ fontWeight: '600', fontSize: '0.95rem', color: '#f8fafc', marginBottom: '4px' }}>
                Drag & drop your WAV audio here
              </p>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.80rem', marginBottom: '14px' }}>
                Standard 16 kHz mono/stereo WAV supported (optimal: 2.0s duration)
              </p>
              <button id="btn-browse-files" className="btn-secondary" type="button">
                Browse Files (.wav)
              </button>
            </div>

            {/* Quick Demo Sample Loaders for Hackathon testing */}
            <div style={{ marginTop: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '6px', marginBottom: '8px', fontSize: '0.76rem', color: 'var(--text-muted)' }}>
                <Sparkles size={13} color="#38bdf8" />
                <span>Quick Test Demo Samples:</span>
              </div>
              <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                <button
                  id="btn-sample-real"
                  className="btn-secondary"
                  style={{
                    fontSize: '0.78rem',
                    padding: '6px 12px',
                    borderColor: 'rgba(16, 185, 129, 0.3)',
                    background: 'rgba(16, 185, 129, 0.06)'
                  }}
                  onClick={() => loadDemoSample('real_0002.wav')}
                >
                  <PlayCircle size={14} color="#10b981" />
                  <span>real_0002.wav (REAL)</span>
                </button>
                <button
                  id="btn-sample-fake"
                  className="btn-secondary"
                  style={{
                    fontSize: '0.78rem',
                    padding: '6px 12px',
                    borderColor: 'rgba(244, 63, 94, 0.3)',
                    background: 'rgba(244, 63, 94, 0.06)'
                  }}
                  onClick={() => loadDemoSample('fake_0001.wav')}
                >
                  <PlayCircle size={14} color="#f43f5e" />
                  <span>fake_0001.wav (SYNTHETIC)</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Tab 2: Microphone Recording */}
        {activeTab === 'record' && !audioFile && (
          <div style={{
            border: '1px solid rgba(56, 189, 248, 0.2)',
            borderRadius: '12px',
            padding: '30px 20px',
            textAlign: 'center',
            background: 'rgba(15, 23, 42, 0.4)',
          }}>
            <div style={{
              width: '60px',
              height: '60px',
              borderRadius: '50%',
              background: isRecording ? 'rgba(244, 63, 94, 0.2)' : 'rgba(56, 189, 248, 0.1)',
              border: `2px solid ${isRecording ? 'var(--fake-crimson)' : 'rgba(56, 189, 248, 0.3)'}`,
              margin: '0 auto 16px auto',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              color: isRecording ? '#f43f5e' : 'var(--accent-cyan)',
              animation: isRecording ? 'pulseGlow 1.5s infinite' : 'none'
            }}>
              <Mic size={30} />
            </div>

            <p style={{ fontWeight: '600', fontSize: '1rem', color: '#f8fafc', marginBottom: '4px' }}>
              {isRecording ? 'Recording in progress...' : 'Record Voice Sample'}
            </p>

            <p style={{ fontFamily: 'var(--font-mono)', fontSize: '1.25rem', color: isRecording ? '#f43f5e' : 'var(--text-muted)', marginBottom: '16px' }}>
              {recordingSeconds.toFixed(1)}s / 5.0s
            </p>

            {!isRecording ? (
              <button id="btn-start-record" className="btn-primary" onClick={startRecording}>
                <Mic size={16} />
                Start Recording
              </button>
            ) : (
              <button
                id="btn-stop-record"
                className="btn-primary"
                onClick={stopRecording}
                style={{ background: 'linear-gradient(135deg, #e11d48 0%, #be123c 100%)', borderColor: 'rgba(244, 63, 94, 0.5)' }}
              >
                <Square size={16} />
                Stop & Save WAV
              </button>
            )}
          </div>
        )}

        {/* Selected Audio File Bar */}
        {audioFile && (
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              padding: '14px 16px',
              borderRadius: '10px',
              background: 'rgba(15, 23, 42, 0.7)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              marginBottom: '18px'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px', minWidth: 0 }}>
                <div style={{
                  width: '40px',
                  height: '40px',
                  borderRadius: '8px',
                  background: 'rgba(56, 189, 248, 0.15)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'var(--accent-cyan)',
                  flexShrink: 0
                }}>
                  <Volume2 size={20} />
                </div>
                <div style={{ overflow: 'hidden' }}>
                  <p id="audio-filename-display" style={{ fontWeight: '600', fontSize: '0.92rem', color: '#f1f5f9', textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                    {audioFile.name || 'Audio Clip'}
                  </p>
                  <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', marginTop: '2px' }}>
                    {formatFileSize(audioFile.size)} • {audioMetadata.duration ? `${audioMetadata.duration.toFixed(2)}s` : 'Reading duration...'} • {audioMetadata.sampleRate ? `${audioMetadata.sampleRate} Hz` : '16 kHz'}
                  </p>
                </div>
              </div>

              <button
                id="btn-clear-audio"
                className="btn-danger"
                onClick={onClearFile}
                disabled={isAnalyzing}
                title="Remove audio"
              >
                <Trash2 size={14} />
                Clear
              </button>
            </div>

            {/* Analyze Action Bar */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <button
                id="btn-analyze-audio"
                className="btn-primary"
                onClick={onAnalyze}
                disabled={isAnalyzing}
                style={{ width: '100%', padding: '14px', fontSize: '1rem' }}
              >
                {isAnalyzing ? (
                  <>
                    <span className="pulse-dot working" />
                    <span>{analysisStage || 'Analyzing audio...'}</span>
                  </>
                ) : (
                  <>
                    <CheckCircle2 size={18} />
                    <span>Analyze Audio</span>
                  </>
                )}
              </button>

              {isAnalyzing && (
                <div style={{
                  padding: '8px 12px',
                  borderRadius: '6px',
                  background: 'rgba(56, 189, 248, 0.08)',
                  border: '1px solid rgba(56, 189, 248, 0.2)',
                  fontSize: '0.78rem',
                  fontFamily: 'var(--font-mono)',
                  color: '#38bdf8',
                  textAlign: 'center'
                }}>
                  Pipeline: {analysisStage}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
