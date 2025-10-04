"""
ANALYSIS MODULE
---------------
Advanced audio analysis tools including spectrum analysis, beat detection, and metering.
"""

import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from scipy import signal


def spectrum_analyzer(y, sr, frame_size=2048, hop_length=512):
    """
    Real-time style spectrum analyzer.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        frame_size (int): FFT frame size
        hop_length (int): Hop length
        
    Returns:
        tuple: (frequencies, times, spectrum)
    """
    # Compute STFT
    D = librosa.stft(y, n_fft=frame_size, hop_length=hop_length)
    magnitude = np.abs(D)
    
    # Convert to dB
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Get frequency and time axes
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), 
                                   sr=sr, hop_length=hop_length)
    
    return frequencies, times, magnitude_db


def beat_detector(y, sr):
    """
    Detect beats and estimate tempo.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        
    Returns:
        dict: Beat information
    """
    # Estimate tempo
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Convert beat frames to time
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Onset detection
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onsets, sr=sr)
    
    return {
        'tempo': tempo,
        'beat_frames': beats,
        'beat_times': beat_times,
        'onset_times': onset_times,
        'onset_envelope': onset_env
    }


def loudness_meter(y, sr, meter_type='rms'):
    """
    Measure loudness over time.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        meter_type (str): 'rms', 'peak', or 'lufs'
        
    Returns:
        dict: Loudness measurements
    """
    frame_length = 2048
    hop_length = 512
    
    if meter_type == 'rms':
        # RMS (Root Mean Square)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)
        
        return {
            'rms': rms,
            'rms_db': rms_db,
            'rms_mean': np.mean(rms_db),
            'rms_max': np.max(rms_db)
        }
    
    elif meter_type == 'peak':
        # Peak levels
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        peak = np.max(np.abs(frames), axis=0)
        peak_db = librosa.amplitude_to_db(peak, ref=1.0)
        
        return {
            'peak': peak,
            'peak_db': peak_db,
            'peak_max': np.max(peak_db),
            'peak_mean': np.mean(peak_db)
        }
    
    elif meter_type == 'lufs':
        # Simplified LUFS calculation
        # (Note: This is a simplified version; proper LUFS requires K-weighting)
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        lufs_approx = -0.691 + 10 * np.log10(np.mean(rms**2))
        
        return {
            'lufs_integrated': lufs_approx,
            'lufs_range': np.max(rms) - np.min(rms)
        }


def frequency_distribution(y, sr, n_bands=10):
    """
    Analyze frequency distribution across bands.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        n_bands (int): Number of frequency bands
        
    Returns:
        dict: Frequency band energies
    """
    # Compute spectrogram
    D = librosa.stft(y)
    magnitude = np.abs(D)
    
    # Define frequency bands (logarithmically spaced)
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)
    
    # Calculate energy in each band
    band_energies = []
    band_names = []
    
    for i in range(n_bands):
        low = band_edges[i]
        high = band_edges[i + 1]
        
        # Find frequency bins in this band
        band_mask = (freq_bins >= low) & (freq_bins < high)
        
        # Sum energy in band
        band_energy = np.sum(magnitude[band_mask, :])
        band_energies.append(band_energy)
        band_names.append(f"{low:.0f}-{high:.0f} Hz")
    
    # Normalize
    band_energies = np.array(band_energies)
    band_energies = band_energies / np.sum(band_energies)
    
    return {
        'band_names': band_names,
        'band_energies': band_energies,
        'band_edges': band_edges
    }


def phase_correlation(y_stereo):
    """
    Calculate phase correlation (stereo compatibility).
    
    Args:
        y_stereo (np.array): Stereo audio (shape: (2, n_samples))
        
    Returns:
        dict: Phase correlation info
    """
    if y_stereo.ndim == 1:
        return {'error': 'Input must be stereo'}
    
    left = y_stereo[0]
    right = y_stereo[1]
    
    # Calculate correlation
    correlation = np.correlate(left, right, mode='valid')[0]
    
    # Normalize by standard deviations
    correlation_normalized = correlation / (np.std(left) * np.std(right) * len(left))
    
    # Calculate phase correlation meter value (-1 to +1)
    mid = (left + right) / 2
    side = (left - right) / 2
    
    energy_mid = np.sum(mid**2)
    energy_side = np.sum(side**2)
    
    if energy_mid + energy_side > 0:
        phase_corr = (energy_mid - energy_side) / (energy_mid + energy_side)
    else:
        phase_corr = 0
    
    return {
        'correlation': correlation_normalized,
        'phase_correlation': phase_corr,
        'mono_compatible': phase_corr > 0
    }


def spectral_features(y, sr):
    """
    Extract spectral features.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        
    Returns:
        dict: Spectral features
    """
    # Spectral centroid (brightness)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    
    # Spectral bandwidth
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    
    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    
    # Spectral contrast
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    
    return {
        'centroid_mean': np.mean(centroid),
        'centroid_std': np.std(centroid),
        'bandwidth_mean': np.mean(bandwidth),
        'bandwidth_std': np.std(bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_std': np.std(rolloff),
        'contrast_mean': np.mean(contrast, axis=1),
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr)
    }


def showcase_analysis(audio_path, output_dir="output", save=True):
    """
    Showcase analysis capabilities.
    
    Args:
        audio_path (str): Path to audio file
        output_dir (str): Output directory
        save (bool): Whether to save visualizations
        
    Returns:
        dict: Analysis results
    """
    print("📊  Audio Analysis Showcase")
    print("=" * 50)
    
    # Load audio
    print("\n📂  Loading audio...")
    y, sr = librosa.load(audio_path, sr=44100, mono=True)
    print(f"   ✓ Loaded {audio_path} ({len(y)/sr:.2f}s)")
    
    results = {}
    
    # 1. Spectrum Analysis
    print("\n🌈  Spectrum analysis...")
    freqs, times, spectrum = spectrum_analyzer(y, sr)
    results['spectrum'] = {'freqs': freqs, 'times': times, 'spectrum': spectrum}
    
    # 2. Beat Detection
    print("\n🥁  Beat detection...")
    beat_info = beat_detector(y, sr)
    results['beats'] = beat_info
    print(f"   ✓ Detected tempo: {beat_info['tempo']:.1f} BPM")
    print(f"   ✓ Found {len(beat_info['beat_times'])} beats")
    
    # 3. Loudness Metering
    print("\n🔊  Loudness metering...")
    rms_info = loudness_meter(y, sr, meter_type='rms')
    peak_info = loudness_meter(y, sr, meter_type='peak')
    results['loudness'] = {'rms': rms_info, 'peak': peak_info}
    print(f"   ✓ RMS mean: {rms_info['rms_mean']:.2f} dB")
    print(f"   ✓ Peak max: {peak_info['peak_max']:.2f} dB")
    
    # 4. Frequency Distribution
    print("\n📊  Frequency distribution...")
    freq_dist = frequency_distribution(y, sr, n_bands=10)
    results['frequency_dist'] = freq_dist
    
    # 5. Spectral Features
    print("\n✨  Spectral features...")
    spectral = spectral_features(y, sr)
    results['spectral'] = spectral
    print(f"   ✓ Spectral centroid: {spectral['centroid_mean']:.0f} Hz")
    print(f"   ✓ Spectral bandwidth: {spectral['bandwidth_mean']:.0f} Hz")
    
    # Visualization
    print("\n📊  Creating visualizations...")
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Waveform
    ax1 = plt.subplot(4, 2, 1)
    librosa.display.waveshow(y, sr=sr, ax=ax1)
    ax1.set_title("Waveform")
    
    # 2. Spectrum
    ax2 = plt.subplot(4, 2, 2)
    librosa.display.specshow(spectrum, sr=sr, x_axis='time', y_axis='hz', ax=ax2)
    ax2.set_title("Spectrogram")
    ax2.set_ylim(0, 8000)
    
    # 3. Beat tracking
    ax3 = plt.subplot(4, 2, 3)
    times = librosa.times_like(beat_info['onset_envelope'], sr=sr)
    ax3.plot(times, beat_info['onset_envelope'], label='Onset Strength')
    ax3.vlines(beat_info['beat_times'], 0, beat_info['onset_envelope'].max(), 
              color='r', alpha=0.5, label='Beats')
    ax3.set_title(f"Beat Tracking (Tempo: {beat_info['tempo']:.1f} BPM)")
    ax3.set_xlabel("Time (s)")
    ax3.legend()
    
    # 4. RMS Loudness
    ax4 = plt.subplot(4, 2, 4)
    frames = range(len(rms_info['rms_db']))
    times_rms = librosa.frames_to_time(frames, sr=sr)
    ax4.plot(times_rms, rms_info['rms_db'], label='RMS')
    ax4.axhline(rms_info['rms_mean'], color='r', linestyle='--', label='Mean')
    ax4.set_title("RMS Loudness")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("dB")
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Frequency Distribution
    ax5 = plt.subplot(4, 2, 5)
    ax5.bar(range(len(freq_dist['band_energies'])), freq_dist['band_energies'])
    ax5.set_xticks(range(len(freq_dist['band_names'])))
    ax5.set_xticklabels(freq_dist['band_names'], rotation=45, ha='right')
    ax5.set_title("Frequency Distribution")
    ax5.set_ylabel("Normalized Energy")
    ax5.grid(True, alpha=0.3, axis='y')
    
    # 6. Spectral Centroid
    ax6 = plt.subplot(4, 2, 6)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    times_spec = librosa.times_like(centroid, sr=sr)
    ax6.plot(times_spec, centroid)
    ax6.set_title("Spectral Centroid (Brightness)")
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Hz")
    ax6.grid(True, alpha=0.3)
    
    # 7. Mel Spectrogram
    ax7 = plt.subplot(4, 2, 7)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel', ax=ax7)
    ax7.set_title("Mel Spectrogram")
    
    # 8. Chromagram
    ax8 = plt.subplot(4, 2, 8)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    librosa.display.specshow(chroma, sr=sr, x_axis='time', y_axis='chroma', ax=ax8)
    ax8.set_title("Chromagram")
    
    plt.tight_layout()
    
    if save:
        plt.savefig(f"{output_dir}/analysis_showcase.png", dpi=150)
        print(f"   ✓ Analysis visualization saved")
    
    plt.show()
    
    return results


if __name__ == "__main__":
    print("Analysis Module - Test Mode")
    print("=" * 50)

