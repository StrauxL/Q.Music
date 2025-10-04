"""
MIXING MODULE
-------------
Multi-track mixing, crossfading, and gain control tools.
"""

import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display


class Mixer:
    """
    Multi-track audio mixer.
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.tracks = {}
        self.track_gains = {}
        self.track_pans = {}
    
    def add_track(self, name, audio, gain_db=0, pan=0.0):
        """
        Add a track to the mixer.
        
        Args:
            name (str): Track name
            audio (np.array): Audio data
            gain_db (float): Track gain in dB
            pan (float): Pan position (-1 = left, 0 = center, 1 = right)
        """
        self.tracks[name] = audio
        self.track_gains[name] = gain_db
        self.track_pans[name] = pan
        print(f"✓ Added track '{name}' ({len(audio)/self.sr:.2f}s, {gain_db:.1f}dB, pan={pan:.2f})")
    
    def set_gain(self, track_name, gain_db):
        """Set track gain in dB."""
        if track_name in self.tracks:
            self.track_gains[track_name] = gain_db
            print(f"✓ Set '{track_name}' gain to {gain_db:.1f} dB")
    
    def set_pan(self, track_name, pan):
        """Set track pan (-1 to 1)."""
        if track_name in self.tracks:
            self.track_pans[track_name] = pan
            print(f"✓ Set '{track_name}' pan to {pan:.2f}")
    
    def apply_gain(self, audio, gain_db):
        """Apply gain in dB to audio."""
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def apply_pan(self, audio, pan):
        """
        Apply panning to mono audio.
        
        Args:
            audio (np.array): Mono audio
            pan (float): Pan position (-1 to 1)
            
        Returns:
            np.array: Stereo audio (shape: (2, n_samples))
        """
        # Constant power panning
        pan_rad = (pan + 1) * np.pi / 4
        left_gain = np.cos(pan_rad)
        right_gain = np.sin(pan_rad)
        
        left = audio * left_gain
        right = audio * right_gain
        
        return np.array([left, right])
    
    def mix(self, normalize=True):
        """
        Mix all tracks together.
        
        Args:
            normalize (bool): Whether to normalize the output
            
        Returns:
            np.array: Mixed stereo audio (shape: (2, n_samples))
        """
        if not self.tracks:
            print("⚠️  No tracks to mix!")
            return None
        
        # Find the longest track
        max_length = max(len(audio) for audio in self.tracks.values())
        
        # Initialize stereo mix
        mixed = np.zeros((2, max_length))
        
        # Mix each track
        for name, audio in self.tracks.items():
            # Apply gain
            gained = self.apply_gain(audio, self.track_gains[name])
            
            # Apply pan (convert to stereo)
            stereo = self.apply_pan(gained, self.track_pans[name])
            
            # Pad if necessary
            if len(audio) < max_length:
                stereo = np.pad(stereo, ((0, 0), (0, max_length - len(audio))), mode='constant')
            
            # Add to mix
            mixed += stereo
        
        # Normalize if requested
        if normalize:
            mixed = mixed / np.max(np.abs(mixed))
        
        print(f"✓ Mixed {len(self.tracks)} tracks into stereo")
        
        return mixed
    
    def visualize(self):
        """Visualize all tracks."""
        n_tracks = len(self.tracks)
        
        if n_tracks == 0:
            print("⚠️  No tracks to visualize!")
            return
        
        fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 2 * n_tracks))
        
        if n_tracks == 1:
            axes = [axes]
        
        for ax, (name, audio) in zip(axes, self.tracks.items()):
            librosa.display.waveshow(audio, sr=self.sr, ax=ax)
            gain = self.track_gains[name]
            pan = self.track_pans[name]
            ax.set_title(f"{name} (Gain: {gain:.1f}dB, Pan: {pan:.2f})")
            ax.set_ylabel("Amplitude")
        
        plt.tight_layout()
        plt.show()


def crossfade(audio1, audio2, crossfade_duration=2.0, sr=44100):
    """
    Crossfade between two audio signals.
    
    Args:
        audio1 (np.array): First audio signal
        audio2 (np.array): Second audio signal
        crossfade_duration (float): Crossfade duration in seconds
        sr (int): Sample rate
        
    Returns:
        np.array: Crossfaded audio
    """
    crossfade_samples = int(crossfade_duration * sr)
    
    # Ensure audio2 is at least as long as crossfade
    if len(audio2) < crossfade_samples:
        audio2 = np.pad(audio2, (0, crossfade_samples - len(audio2)), mode='constant')
    
    # Trim audio1 if needed
    if len(audio1) < crossfade_samples:
        # Too short, just concatenate
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    fade_out = np.linspace(1, 0, crossfade_samples)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Split audio1
    audio1_pre = audio1[:-crossfade_samples]
    audio1_fade = audio1[-crossfade_samples:]
    
    # Split audio2
    audio2_fade = audio2[:crossfade_samples]
    audio2_post = audio2[crossfade_samples:]
    
    # Apply fades
    audio1_faded = audio1_fade * fade_out
    audio2_faded = audio2_fade * fade_in
    
    # Mix crossfade region
    crossfaded_region = audio1_faded + audio2_faded
    
    # Concatenate all parts
    result = np.concatenate([audio1_pre, crossfaded_region, audio2_post])
    
    return result


def auto_duck(main_audio, ducking_audio, threshold_db=-20, ratio=3, sr=44100):
    """
    Auto-ducking: reduce main_audio volume when ducking_audio is present.
    (e.g., lower music when voice is present)
    
    Args:
        main_audio (np.array): Main audio (to be ducked)
        ducking_audio (np.array): Control audio (triggers ducking)
        threshold_db (float): Threshold in dB
        ratio (float): Ducking ratio
        sr (int): Sample rate
        
    Returns:
        np.array: Ducked audio
    """
    # Make sure both audios are same length
    max_len = max(len(main_audio), len(ducking_audio))
    main_audio = np.pad(main_audio, (0, max_len - len(main_audio)), mode='constant')
    ducking_audio = np.pad(ducking_audio, (0, max_len - len(ducking_audio)), mode='constant')
    
    # Calculate envelope of ducking signal
    envelope = np.abs(ducking_audio)
    
    # Smooth envelope
    from scipy.ndimage import uniform_filter1d
    window_size = int(0.01 * sr)  # 10ms window
    envelope_smooth = uniform_filter1d(envelope, size=window_size)
    
    # Convert to dB
    envelope_db = 20 * np.log10(envelope_smooth + 1e-10)
    
    # Calculate ducking gain
    threshold = threshold_db
    gain_reduction_db = np.where(
        envelope_db > threshold,
        (envelope_db - threshold) / ratio,
        0
    )
    
    # Convert to linear gain
    ducking_gain = 10 ** (-gain_reduction_db / 20)
    
    # Apply ducking
    ducked = main_audio * ducking_gain
    
    return ducked


def create_stereo_from_mono(audio, width=1.0, delay_ms=10):
    """
    Create stereo from mono using Haas effect.
    
    Args:
        audio (np.array): Mono audio
        width (float): Stereo width (0-1)
        delay_ms (float): Delay between channels in milliseconds
        
    Returns:
        np.array: Stereo audio (shape: (2, n_samples))
    """
    # Create delay
    delay_samples = int(delay_ms * 44100 / 1000)
    
    # Create left and right channels
    left = audio
    right = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
    
    # Apply width
    mid = (left + right) / 2
    side = (left - right) / 2
    
    left_out = mid + side * width
    right_out = mid - side * width
    
    return np.array([left_out, right_out])


def showcase_mixing(audio_path1, audio_path2=None, output_dir="output", save=True):
    """
    Showcase mixing capabilities.
    
    Args:
        audio_path1 (str): Path to first audio file
        audio_path2 (str): Path to second audio file (optional)
        output_dir (str): Output directory
        save (bool): Whether to save files
        
    Returns:
        dict: Dictionary of mixed results
    """
    print("🎚️  Mixing Showcase")
    print("=" * 50)
    
    results = {}
    
    # Load audio
    print("\n📂  Loading audio files...")
    y1, sr = librosa.load(audio_path1, sr=44100, mono=True)
    print(f"   ✓ Loaded {audio_path1}")
    
    if audio_path2:
        y2, _ = librosa.load(audio_path2, sr=44100, mono=True)
        print(f"   ✓ Loaded {audio_path2}")
    else:
        # Create a second track by pitch shifting
        y2 = librosa.effects.pitch_shift(y1, sr=sr, n_steps=7)
        print(f"   ✓ Created second track (pitch shifted)")
    
    # Ensure same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # 1. Multi-track mixing
    print("\n🎛️  Multi-track mixing demo...")
    mixer = Mixer(sr)
    mixer.add_track("Track 1", y1, gain_db=0, pan=-0.5)
    mixer.add_track("Track 2", y2, gain_db=-3, pan=0.5)
    
    mixed_stereo = mixer.mix(normalize=True)
    results['multitrack_mix'] = mixed_stereo
    
    if save:
        # Save as stereo
        sf.write(f"{output_dir}/mix_multitrack.wav", mixed_stereo.T, sr)
        print(f"   ✓ Multi-track mix saved")
    
    # 2. Crossfade
    print("\n🔀  Crossfade demo...")
    crossfaded = crossfade(y1, y2, crossfade_duration=2.0, sr=sr)
    results['crossfade'] = crossfaded
    
    if save:
        sf.write(f"{output_dir}/mix_crossfade.wav", crossfaded, sr)
        print(f"   ✓ Crossfade saved")
    
    # 3. Auto-ducking
    print("\n🎤  Auto-ducking demo...")
    ducked = auto_duck(y1, y2, threshold_db=-25, ratio=4, sr=sr)
    results['auto_duck'] = ducked
    
    if save:
        sf.write(f"{output_dir}/mix_auto_duck.wav", ducked, sr)
        print(f"   ✓ Auto-ducked audio saved")
    
    # 4. Stereo from mono
    print("\n🎧  Stereo from mono demo...")
    stereo = create_stereo_from_mono(y1, width=0.8, delay_ms=15)
    results['stereo_haas'] = stereo
    
    if save:
        sf.write(f"{output_dir}/mix_stereo_haas.wav", stereo.T, sr)
        print(f"   ✓ Stereo (Haas effect) saved")
    
    # Visualize
    print("\n📊  Creating visualization...")
    plt.figure(figsize=(14, 10))
    
    # Original tracks
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y1, sr=sr)
    plt.title("Track 1")
    
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(y2, sr=sr)
    plt.title("Track 2")
    
    # Mixed
    plt.subplot(4, 1, 3)
    librosa.display.waveshow(mixed_stereo[0], sr=sr, label='Left', alpha=0.7)
    librosa.display.waveshow(mixed_stereo[1], sr=sr, label='Right', alpha=0.7)
    plt.title("Multi-track Mix (Stereo)")
    plt.legend()
    
    # Crossfade
    plt.subplot(4, 1, 4)
    librosa.display.waveshow(crossfaded, sr=sr)
    plt.title("Crossfade")
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/mix_comparison.png", dpi=150)
        print(f"   ✓ Visualization saved")
    plt.show()
    
    return results


if __name__ == "__main__":
    print("Mixing Module - Test Mode")
    print("=" * 50)

