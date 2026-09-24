import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import AudioInput from './components/AudioInput';
import AudioPlayer from './components/AudioPlayer';
import DetectionResult from './components/DetectionResult';
import WaveformViewer from './components/WaveformViewer';
import SpectrogramViewer from './components/SpectrogramViewer';
import SegmentAnalysis from './components/SegmentAnalysis';
import ModelSpecsCard from './components/ModelSpecsCard';
import HowItWorks from './components/HowItWorks';

import { checkBackendHealth, predictAudio } from './services/api';
import { decodeAudioData } from './utils/audioHelper';

export default function App() {
  // Backend Status
  const [backendStatus, setBackendStatus] = useState({
    online: false,
    status: 'checking',
    modelLoaded: false,
    error: null
  });

  // Audio State
  const [audioFile, setAudioFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [audioBuffer, setAudioBuffer] = useState(null);
  const [audioMetadata, setAudioMetadata] = useState({ duration: 0, sampleRate: 0, channels: 0 });

  // Playback State
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);
  const [volume, setVolume] = useState(0.85);
  const [isMuted, setIsMuted] = useState(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisStage, setAnalysisStage] = useState('');
  const [detectionResult, setDetectionResult] = useState(null);
  const [error, setError] = useState(null);

  // Audio Element ref for HTML5 playback
  const audioRef = useRef(null);

  // --------------------------------------------------------------------------
  // Periodic Backend Health Polling (every 6 seconds)
  // --------------------------------------------------------------------------
  useEffect(() => {
    let isMounted = true;

    const pollHealth = async () => {
      const status = await checkBackendHealth();
      if (isMounted) {
        setBackendStatus(status);
      }
    };

    pollHealth();
    const interval = setInterval(pollHealth, 6000);

    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // --------------------------------------------------------------------------
  // Audio Playback Tracking
  // --------------------------------------------------------------------------
  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;

    const updateTime = () => setCurrentTime(audio.currentTime);
    const handleEnded = () => {
      setIsPlaying(false);
      setCurrentTime(audio.duration || 0);
    };
    const handleLoadedMetadata = () => {
      setDuration(audio.duration || 0);
    };

    audio.addEventListener('timeupdate', updateTime);
    audio.addEventListener('ended', handleEnded);
    audio.addEventListener('loadedmetadata', handleLoadedMetadata);

    return () => {
      audio.removeEventListener('timeupdate', updateTime);
      audio.removeEventListener('ended', handleEnded);
      audio.removeEventListener('loadedmetadata', handleLoadedMetadata);
    };
  }, [audioUrl]);

  // Sync volume with audio element
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.volume = isMuted ? 0 : volume;
    }
  }, [volume, isMuted]);

  // --------------------------------------------------------------------------
  // File Selection & Audio Decoding
  // --------------------------------------------------------------------------
  const handleSelectFile = async (file, filename) => {
    try {
      setError(null);
      setDetectionResult(null);
      setIsPlaying(false);
      setCurrentTime(0);

      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }

      const url = URL.createObjectURL(file);
      setAudioFile(file);
      setAudioUrl(url);

      // Decode AudioBuffer for Canvas visualizers
      const decodedBuffer = await decodeAudioData(file);
      setAudioBuffer(decodedBuffer);
      setDuration(decodedBuffer.duration);
      setAudioMetadata({
        duration: decodedBuffer.duration,
        sampleRate: decodedBuffer.sampleRate,
        channels: decodedBuffer.numberOfChannels
      });

    } catch (err) {
      setError(`Failed to load audio file: ${err.message}`);
    }
  };

  const handleClearFile = () => {
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }
    if (audioUrl) {
      URL.revokeObjectURL(audioUrl);
    }
    setAudioFile(null);
    setAudioUrl(null);
    setAudioBuffer(null);
    setAudioMetadata({ duration: 0, sampleRate: 0, channels: 0 });
    setDetectionResult(null);
    setIsPlaying(false);
    setCurrentTime(0);
    setDuration(0);
    setError(null);
  };

  // --------------------------------------------------------------------------
  // Audio Playback Controls
  // --------------------------------------------------------------------------
  const handlePlayToggle = () => {
    if (!audioRef.current || !audioUrl) return;

    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      // If at end, start from beginning
      if (currentTime >= duration && duration > 0) {
        audioRef.current.currentTime = 0;
        setCurrentTime(0);
      }
      audioRef.current.play().then(() => {
        setIsPlaying(true);
      }).catch(err => {
        console.warn('Playback error:', err);
      });
    }
  };

  const handleSeek = (timeSecs) => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = timeSecs;
    setCurrentTime(timeSecs);
  };

  const handleRestart = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
    audioRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
  };

  // --------------------------------------------------------------------------
  // Deepfake Analysis Trigger
  // --------------------------------------------------------------------------
  const handleAnalyze = async () => {
    if (!audioFile) {
      setError('Please select or record a WAV audio file first.');
      return;
    }

    if (!backendStatus.online) {
      setError('FastAPI backend is offline. Please ensure the server is running on http://127.0.0.1:8000');
      return;
    }

    setIsAnalyzing(true);
    setError(null);

    // Progressive UI feedback stages
    setAnalysisStage('Analyzing audio...');
    const stageTimer1 = setTimeout(() => {
      setAnalysisStage('Extracting spectral features...');
    }, 250);
    const stageTimer2 = setTimeout(() => {
      setAnalysisStage('Running CNN-BiLSTM analysis...');
    }, 500);

    try {
      const result = await predictAudio(audioFile, audioFile.name || 'sample.wav');
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      setDetectionResult(result);
    } catch (err) {
      clearTimeout(stageTimer1);
      clearTimeout(stageTimer2);
      setError(`Detection Error: ${err.message || 'Failed to analyze audio'}`);
    } finally {
      setIsAnalyzing(false);
      setAnalysisStage('');
    }
  };

  return (
    <div className="app-container">
      {/* Hidden HTML5 Audio Element */}
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          preload="auto"
        />
      )}

      {/* Header */}
      <Header backendStatus={backendStatus} />

      {/* Main Forensic Grid: Audio Input & Detection Result */}
      <div className="dashboard-grid">
        {/* Left Column: Audio Input & Player */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <AudioInput
            audioFile={audioFile}
            audioMetadata={audioMetadata}
            onSelectFile={handleSelectFile}
            onClearFile={handleClearFile}
            onAnalyze={handleAnalyze}
            isAnalyzing={isAnalyzing}
            analysisStage={analysisStage}
            error={error}
            onError={setError}
          />

          {audioFile && (
            <AudioPlayer
              audioUrl={audioUrl}
              isPlaying={isPlaying}
              currentTime={currentTime}
              duration={duration}
              volume={volume}
              isMuted={isMuted}
              onPlayToggle={handlePlayToggle}
              onSeek={handleSeek}
              onVolumeChange={setVolume}
              onMuteToggle={() => setIsMuted(!isMuted)}
              onRestart={handleRestart}
            />
          )}
        </div>

        {/* Right Column: Large Detection Result Card */}
        <div>
          <DetectionResult result={detectionResult} />
        </div>
      </div>

      {/* Visualizers Grid: Waveform & Spectral Heatmap */}
      <div className="dashboard-grid">
        <WaveformViewer
          audioBuffer={audioBuffer}
          currentTime={currentTime}
          duration={duration}
          onSeek={handleSeek}
        />

        <SpectrogramViewer
          audioBuffer={audioBuffer}
          currentTime={currentTime}
          duration={duration}
        />
      </div>

      {/* Segment Level Analysis (Full Width) */}
      <SegmentAnalysis
        audioBuffer={audioBuffer}
        result={detectionResult}
      />

      {/* Architecture Specs & Forensics Methodology */}
      <div className="dashboard-grid">
        <ModelSpecsCard />
        <HowItWorks />
      </div>
    </div>
  );
}
