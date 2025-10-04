"""
ANALYSIS MODULE
---------------
Advanced audio analysis tools including spectrum analysis, beat detection, and metering.

This module provides beginner-friendly functions to analyze audio files and understand
their properties like frequency content, rhythm, loudness, and more.

Key Concepts for Beginners:
- Spectrum Analysis: Shows which frequencies are present in audio
- Beat Detection: Finds the rhythm and tempo of music
- Loudness Metering: Measures how loud different parts of audio are
- Frequency Distribution: Shows energy across different frequency bands
- Phase Correlation: Measures stereo compatibility

Author: Q.Wave Team
"""

# Import necessary libraries for audio processing and visualization
import numpy as np          # For numerical operations and arrays
import librosa             # Main audio processing library
import librosa.display     # For audio visualization
import matplotlib.pyplot as plt  # For creating plots and graphs
from scipy import signal   # For signal processing functions


def spectrum_analyzer(y, sr, frame_size=2048, hop_length=512):
    """
    Real-time style spectrum analyzer.
    
    This function analyzes the frequency content of audio over time, similar to
    what you see in audio software like Audacity or professional audio analyzers.
    
    What it does:
    1. Takes small chunks of audio (frames) and analyzes their frequency content
    2. Shows which frequencies are present and how strong they are
    3. Creates a "spectrogram" - a visual representation of frequencies over time
    
    Args:
        y (np.array): Audio data (the actual sound samples)
        sr (int): Sample rate (how many samples per second, usually 44100 Hz)
        frame_size (int): Size of each analysis window (2048 samples = ~46ms at 44.1kHz)
        hop_length (int): How much to move forward between frames (512 samples = ~12ms)
        
    Returns:
        tuple: (frequencies, times, spectrum)
            - frequencies: Array of frequency values in Hz
            - times: Array of time points in seconds
            - spectrum: 2D array showing frequency strength over time (in dB)
    """
    # Step 1: Compute Short-Time Fourier Transform (STFT)
    # This breaks the audio into small time windows and analyzes each window's frequencies
    D = librosa.stft(y, n_fft=frame_size, hop_length=hop_length)
    
    # Step 2: Get the magnitude (strength) of each frequency
    # The STFT gives complex numbers, we only need the magnitude (how strong each frequency is)
    magnitude = np.abs(D)
    
    # Step 3: Convert to decibels (dB) for better visualization
    # Human hearing is logarithmic, so dB scale is more natural
    # ref=np.max means we use the maximum value as our reference point (0 dB)
    magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)
    
    # Step 4: Create the frequency axis (which frequencies we're analyzing)
    # This tells us what frequency each row in our spectrum represents
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)
    
    # Step 5: Create the time axis (when each analysis window occurred)
    # This tells us what time each column in our spectrum represents
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), 
                                   sr=sr, hop_length=hop_length)
    
    return frequencies, times, magnitude_db


def beat_detector(y, sr):
    """
    Detect beats and estimate tempo.
    
    This function finds the rhythm and tempo of music, similar to how DJ software
    automatically detects the BPM (Beats Per Minute) of songs.
    
    What it does:
    1. Analyzes the audio to find rhythmic patterns
    2. Estimates the tempo (speed) of the music
    3. Finds individual beat locations
    4. Detects "onsets" - moments when new sounds start
    
    Args:
        y (np.array): Audio data (the actual sound samples)
        sr (int): Sample rate (how many samples per second, usually 44100 Hz)
        
    Returns:
        dict: Beat information containing:
            - tempo: Beats per minute (BPM) - how fast the music is
            - beat_frames: Frame numbers where beats occur
            - beat_times: Time in seconds where beats occur
            - onset_times: Time in seconds where new sounds start
            - onset_envelope: Strength of onsets over time
    """
    # Step 1: Estimate tempo and find beat locations
    # This is the main beat tracking algorithm that finds the rhythm
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    
    # Step 2: Convert tempo to a simple number if it's an array
    # Sometimes librosa returns arrays, we want just the number
    if isinstance(tempo, np.ndarray):
        tempo = tempo.item()
    
    # Step 3: Convert beat frame numbers to actual time in seconds
    # Beats are initially found as frame numbers, we convert to seconds for easier use
    beat_times = librosa.frames_to_time(beats, sr=sr)
    
    # Step 4: Detect onsets (when new sounds start)
    # Onsets are different from beats - they detect any new sound, not just the main rhythm
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)  # Calculate onset strength over time
    onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)  # Find onset locations
    onset_times = librosa.frames_to_time(onsets, sr=sr)  # Convert to seconds
    
    return {
        'tempo': tempo,                    # BPM (e.g., 120 BPM)
        'beat_frames': beats,              # Frame numbers of beats
        'beat_times': beat_times,          # Time in seconds of beats
        'onset_times': onset_times,        # Time in seconds of onsets
        'onset_envelope': onset_env        # Onset strength over time
    }


def loudness_meter(y, sr, meter_type='rms'):
    """
    Measure loudness over time.
    
    This function measures how loud different parts of audio are, similar to
    the volume meters you see in audio software or on mixing boards.
    
    What it does:
    1. Analyzes audio in small chunks over time
    2. Calculates different types of loudness measurements
    3. Returns statistics about the audio's volume levels
    
    Types of loudness measurements:
    - RMS: Average loudness (like how loud it "feels" to humans)
    - Peak: Maximum instantaneous loudness (loudest moments)
    - LUFS: Professional loudness standard (used in broadcasting)
    
    Args:
        y (np.array): Audio data (the actual sound samples)
        sr (int): Sample rate (how many samples per second, usually 44100 Hz)
        meter_type (str): Type of measurement - 'rms', 'peak', or 'lufs'
        
    Returns:
        dict: Loudness measurements containing various statistics
    """
    # Set up analysis parameters
    frame_length = 2048  # Size of each analysis window (about 46ms at 44.1kHz)
    hop_length = 512     # How much to move forward between windows (about 12ms)
    
    if meter_type == 'rms':
        # RMS (Root Mean Square) - measures average loudness
        # This is closest to how humans perceive loudness
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        rms_db = librosa.amplitude_to_db(rms, ref=1.0)  # Convert to decibels
        
        return {
            'rms': rms,                    # RMS values over time
            'rms_db': rms_db,              # RMS in decibels
            'rms_mean': np.mean(rms_db),   # Average RMS level
            'rms_max': np.max(rms_db)      # Maximum RMS level
        }
    
    elif meter_type == 'peak':
        # Peak levels - measures the loudest instantaneous moments
        # Useful for preventing clipping and distortion
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        peak = np.max(np.abs(frames), axis=0)  # Find maximum absolute value in each frame
        peak_db = librosa.amplitude_to_db(peak, ref=1.0)  # Convert to decibels
        
        return {
            'peak': peak,                  # Peak values over time
            'peak_db': peak_db,            # Peak levels in decibels
            'peak_max': np.max(peak_db),   # Maximum peak level
            'peak_mean': np.mean(peak_db)  # Average peak level
        }
    
    elif meter_type == 'lufs':
        # LUFS (Loudness Units relative to Full Scale) - professional loudness standard
        # Used in broadcasting and streaming (like Netflix, Spotify)
        # Note: This is a simplified version; proper LUFS requires K-weighting
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
        lufs_approx = -0.691 + 10 * np.log10(np.mean(rms**2))  # Simplified LUFS calculation
        
        return {
            'lufs_integrated': lufs_approx,           # Overall loudness in LUFS
            'lufs_range': np.max(rms) - np.min(rms)  # Dynamic range
        }


def frequency_distribution(y, sr, n_bands=10):
    """
    Analyze frequency distribution across bands.
    
    This function divides the audio spectrum into frequency bands (like bass, mid, treble)
    and shows how much energy is in each band. It's like an EQ analyzer that shows
    which frequency ranges are most prominent in the audio.
    
    What it does:
    1. Analyzes the frequency content of the audio
    2. Divides frequencies into bands (like bass, midrange, treble)
    3. Calculates how much energy is in each band
    4. Returns normalized results showing the balance of frequencies
    
    Args:
        y (np.array): Audio data (the actual sound samples)
        sr (int): Sample rate (how many samples per second, usually 44100 Hz)
        n_bands (int): Number of frequency bands to divide into (default: 10)
        
    Returns:
        dict: Frequency band analysis containing:
            - band_names: Names of each frequency band (e.g., "20-200 Hz")
            - band_energies: Normalized energy in each band (0-1)
            - band_edges: Frequency boundaries of each band
    """
    # Step 1: Compute spectrogram (frequency analysis over time)
    D = librosa.stft(y)  # Short-Time Fourier Transform
    magnitude = np.abs(D)  # Get magnitude (strength) of each frequency
    
    # Step 2: Define frequency bands using logarithmic spacing
    # Human hearing is logarithmic, so we use log spacing for more natural bands
    freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)  # All frequency bins
    band_edges = np.logspace(np.log10(20), np.log10(sr/2), n_bands + 1)  # Band boundaries
    
    # Step 3: Calculate energy in each frequency band
    band_energies = []  # Will store energy for each band
    band_names = []     # Will store names for each band
    
    for i in range(n_bands):
        low = band_edges[i]      # Lower frequency of this band
        high = band_edges[i + 1] # Upper frequency of this band
        
        # Find which frequency bins belong to this band
        band_mask = (freq_bins >= low) & (freq_bins < high)
        
        # Sum up all the energy in this frequency band
        band_energy = np.sum(magnitude[band_mask, :])
        band_energies.append(band_energy)
        band_names.append(f"{low:.0f}-{high:.0f} Hz")  # Create readable band name
    
    # Step 4: Normalize the energies so they sum to 1
    # This makes it easy to see the relative importance of each band
    band_energies = np.array(band_energies)
    band_energies = band_energies / np.sum(band_energies)
    
    return {
        'band_names': band_names,      # Names like "20-200 Hz", "200-2000 Hz", etc.
        'band_energies': band_energies, # Normalized energy (0-1) in each band
        'band_edges': band_edges       # Frequency boundaries of each band
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

