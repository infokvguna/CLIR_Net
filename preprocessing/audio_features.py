"""
Audio Feature Extraction Module
Extracts waveforms, mel-spectrograms, and prosodic features (F0, energy, jitter, shimmer, HNR)
"""

import numpy as np
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """Extract comprehensive audio features for emotion recognition"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length_ms: float = 10.0,
        win_length_ms: float = 25.0,
        f0_min: float = 75.0,
        f0_max: float = 600.0
    ):
        self.sr = sample_rate
        self.n_mels = n_mels
        self.hop_length = int(hop_length_ms * sample_rate / 1000)
        self.win_length = int(win_length_ms * sample_rate / 1000)
        self.n_fft = self.win_length
        self.f0_min = f0_min
        self.f0_max = f0_max
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load audio file and resample to target sample rate"""
        y, sr = librosa.load(audio_path, sr=self.sr)
        return y
    
    def extract_mel_spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram"""
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr // 2
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db.astype(np.float32)
    
    def extract_f0(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract fundamental frequency (F0) using librosa pyin
        Returns: f0 values and voiced probability
        """
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sr,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_probs = np.nan_to_num(voiced_probs, nan=0.0)
        return f0.astype(np.float32), voiced_probs.astype(np.float32)
    
    def extract_energy(self, y: np.ndarray) -> np.ndarray:
        """Extract RMS energy"""
        rms = librosa.feature.rms(
            y=y,
            frame_length=self.win_length,
            hop_length=self.hop_length
        )[0]
        return rms.astype(np.float32)
    
    def extract_jitter_shimmer(self, y: np.ndarray) -> Dict[str, float]:
        """
        Extract jitter and shimmer using Parselmouth (Praat)
        These are utterance-level features
        """
        try:
            # Create Praat Sound object
            snd = parselmouth.Sound(y, sampling_frequency=self.sr)
            
            # Extract pitch for jitter calculation
            pitch = snd.to_pitch(time_step=0.01, pitch_floor=self.f0_min, pitch_ceiling=self.f0_max)
            
            # Point process for jitter
            point_process = call(snd, "To PointProcess (periodic, cc)", self.f0_min, self.f0_max)
            
            # Jitter (local)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Jitter (rap)
            jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            
            # Shimmer requires intensity
            # Shimmer (local)
            shimmer_local = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Shimmer (apq3)
            shimmer_apq3 = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Harmonics-to-Noise Ratio
            harmonicity = call(snd, "To Harmonicity (cc)", 0.01, self.f0_min, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            
            return {
                'jitter_local': float(jitter_local) if not np.isnan(jitter_local) else 0.0,
                'jitter_rap': float(jitter_rap) if not np.isnan(jitter_rap) else 0.0,
                'shimmer_local': float(shimmer_local) if not np.isnan(shimmer_local) else 0.0,
                'shimmer_apq3': float(shimmer_apq3) if not np.isnan(shimmer_apq3) else 0.0,
                'hnr': float(hnr) if not np.isnan(hnr) else 0.0
            }
        except Exception as e:
            print(f"Warning: Could not extract jitter/shimmer: {e}")
            return {
                'jitter_local': 0.0,
                'jitter_rap': 0.0,
                'shimmer_local': 0.0,
                'shimmer_apq3': 0.0,
                'hnr': 0.0
            }
    
    def extract_spectral_features(self, y: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract spectral features (centroid, rolloff, flux)"""
        spectral_centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        
        spectral_flux = librosa.onset.onset_strength(
            y=y, sr=self.sr, hop_length=self.hop_length
        )
        
        return {
            'spectral_centroid': spectral_centroid.astype(np.float32),
            'spectral_rolloff': spectral_rolloff.astype(np.float32),
            'spectral_flux': spectral_flux.astype(np.float32)
        }
    
    def extract_all_features(self, audio_path: str) -> Dict[str, np.ndarray]:
        """
        Extract all audio features from an audio file
        
        Returns:
            Dictionary containing:
            - waveform: (T,) raw audio
            - mel_db: (n_mels, T_frames) mel-spectrogram in dB
            - prosody: (n_prosody_features, T_frames) frame-level prosody
            - prosody_global: dict of utterance-level prosody features
        """
        # Load audio
        y = self.load_audio(audio_path)
        
        # Extract features
        mel_db = self.extract_mel_spectrogram(y)
        f0, voiced_prob = self.extract_f0(y)
        energy = self.extract_energy(y)
        spectral = self.extract_spectral_features(y)
        jitter_shimmer = self.extract_jitter_shimmer(y)
        
        # Ensure all frame-level features have same length
        min_frames = min(
            mel_db.shape[1],
            len(f0),
            len(energy),
            len(spectral['spectral_centroid'])
        )
        
        # Truncate to minimum length
        mel_db = mel_db[:, :min_frames]
        f0 = f0[:min_frames]
        voiced_prob = voiced_prob[:min_frames]
        energy = energy[:min_frames]
        
        # Stack frame-level prosodic features
        # Shape: (n_features, T_frames)
        prosody_frames = np.vstack([
            f0,
            voiced_prob,
            energy,
            spectral['spectral_centroid'][:min_frames],
            spectral['spectral_rolloff'][:min_frames],
            spectral['spectral_flux'][:min_frames]
        ])
        
        return {
            'waveform': y.astype(np.float32),
            'mel_db': mel_db,
            'prosody_frames': prosody_frames.astype(np.float32),
            'prosody_global': jitter_shimmer,
            'sample_rate': self.sr,
            'duration': len(y) / self.sr
        }


def test_extractor():
    """Test the audio feature extractor"""
    import matplotlib.pyplot as plt
    
    # Create a test signal (sine wave with varying frequency)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    # Simulate emotional speech with varying pitch
    f0_contour = 200 + 50 * np.sin(2 * np.pi * 2 * t)
    y = np.sin(2 * np.pi * f0_contour * t)
    
    # Save test audio
    test_path = "test_audio.wav"
    sf.write(test_path, y, sr)
    
    # Extract features
    extractor = AudioFeatureExtractor(sample_rate=sr)
    features = extractor.extract_all_features(test_path)
    
    print("Extracted features:")
    print(f"  Waveform shape: {features['waveform'].shape}")
    print(f"  Mel-spectrogram shape: {features['mel_db'].shape}")
    print(f"  Prosody frames shape: {features['prosody_frames'].shape}")
    print(f"  Global prosody: {features['prosody_global']}")
    print(f"  Duration: {features['duration']:.2f}s")
    
    # Visualize
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    # Waveform
    axes[0].plot(features['waveform'])
    axes[0].set_title('Waveform')
    axes[0].set_xlabel('Sample')
    
    # Mel-spectrogram
    librosa.display.specshow(
        features['mel_db'],
        sr=sr,
        hop_length=extractor.hop_length,
        x_axis='time',
        y_axis='mel',
        ax=axes[1]
    )
    axes[1].set_title('Mel-Spectrogram (dB)')
    
    # Prosody
    time_frames = np.arange(features['prosody_frames'].shape[1]) * extractor.hop_length / sr
    axes[2].plot(time_frames, features['prosody_frames'][0], label='F0')
    axes[2].plot(time_frames, features['prosody_frames'][2] * 100, label='Energy (scaled)')
    axes[2].set_title('Prosodic Features')
    axes[2].set_xlabel('Time (s)')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('audio_features_test.png', dpi=150)
    print("Saved visualization to audio_features_test.png")


if __name__ == "__main__":
    test_extractor()
