"""
MIXING MODULE
-------------
Multi-track mixing, crossfading, and gain control tools.

This module provides beginner-friendly functions for mixing multiple audio tracks
together, creating smooth transitions between songs, and controlling volume levels.
It's perfect for learning how professional audio mixing works.

Key Concepts for Beginners:
- Multi-track Mixing: Combining multiple audio tracks into one final mix
- Crossfading: Smoothly transitioning between two audio tracks
- Gain Control: Adjusting the volume of individual tracks
- Panning: Positioning sounds in the stereo field (left/right)
- Auto-ducking: Automatically lowering one track when another is playing
- Stereo Widening: Making mono audio sound wider in stereo

Author: Q.Wave Team
"""

# Import necessary libraries for audio processing and visualization
import numpy as np         # For numerical operations and arrays
import librosa            # Main audio processing library
import soundfile as sf    # For reading and writing audio files
import matplotlib.pyplot as plt  # For creating plots and graphs
import librosa.display    # For audio visualization


class Mixer:
    """
    Multi-track audio mixer.
    
    A mixer is like a mixing board in a recording studio - it allows you to
    combine multiple audio tracks, control their volume levels, and position
    them in the stereo field (left/right). It's essential for creating
    professional-sounding audio mixes.
    
    This mixer can:
    - Add multiple audio tracks
    - Control the volume (gain) of each track
    - Position tracks in the stereo field (panning)
    - Mix all tracks together into a final stereo output
    - Visualize all tracks for easy comparison
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the mixer.
        
        Args:
            sample_rate (int): How many samples per second (usually 44100 Hz)
        """
        self.sr = sample_rate        # Store the sample rate
        self.tracks = {}             # Dictionary to store audio tracks
        self.track_gains = {}        # Dictionary to store track volume levels
        self.track_pans = {}         # Dictionary to store track pan positions
    
    def add_track(self, name, audio, gain_db=0, pan=0.0):
        """
        Add a track to the mixer.
        
        This adds an audio track to the mixer with specified volume and pan settings.
        Think of it like adding a new instrument to your mix.
        
        Args:
            name (str): Track name (e.g., "vocals", "guitar", "drums")
            audio (np.array): Audio data (the actual sound samples)
            gain_db (float): Track gain in dB (0 = normal, positive = louder, negative = quieter)
            pan (float): Pan position (-1 = left, 0 = center, 1 = right)
        """
        # Store the track and its settings
        self.tracks[name] = audio
        self.track_gains[name] = gain_db
        self.track_pans[name] = pan
        
        # Print confirmation with track details
        print(f"✓ Added track '{name}' ({len(audio)/self.sr:.2f}s, {gain_db:.1f}dB, pan={pan:.2f})")
    
    def set_gain(self, track_name, gain_db):
        """
        Set track gain in dB.
        
        This changes the volume of an existing track.
        
        Args:
            track_name (str): Name of the track to adjust
            gain_db (float): New gain in dB (0 = normal, positive = louder, negative = quieter)
        """
        if track_name in self.tracks:
            self.track_gains[track_name] = gain_db
            print(f"✓ Set '{track_name}' gain to {gain_db:.1f} dB")
    
    def set_pan(self, track_name, pan):
        """
        Set track pan (-1 to 1).
        
        This changes the stereo position of an existing track.
        
        Args:
            track_name (str): Name of the track to adjust
            pan (float): Pan position (-1 = left, 0 = center, 1 = right)
        """
        if track_name in self.tracks:
            self.track_pans[track_name] = pan
            print(f"✓ Set '{track_name}' pan to {pan:.2f}")
    
    def apply_gain(self, audio, gain_db):
        """
        Apply gain in dB to audio.
        
        This converts dB gain to linear gain and applies it to the audio.
        
        Args:
            audio (np.array): Audio data
            gain_db (float): Gain in dB
            
        Returns:
            np.array: Audio with gain applied
        """
        # Convert dB to linear gain
        gain_linear = 10 ** (gain_db / 20)
        return audio * gain_linear
    
    def apply_pan(self, audio, pan):
        """
        Apply panning to mono audio.
        
        This converts mono audio to stereo and positions it in the stereo field.
        It uses constant power panning to maintain consistent volume levels.
        
        Args:
            audio (np.array): Mono audio data
            pan (float): Pan position (-1 to 1)
            
        Returns:
            np.array: Stereo audio (shape: (2, n_samples))
        """
        # Constant power panning - maintains consistent volume across the stereo field
        pan_rad = (pan + 1) * np.pi / 4  # Convert pan to radians
        left_gain = np.cos(pan_rad)      # Left channel gain
        right_gain = np.sin(pan_rad)     # Right channel gain
        
        # Apply gains to create stereo
        left = audio * left_gain
        right = audio * right_gain
        
        return np.array([left, right])
    
    def mix(self, normalize=True):
        """
        Mix all tracks together.
        
        This combines all the tracks into a final stereo mix. It's like
        the final step in a recording studio where all the tracks are
        combined into the final song.
        
        Args:
            normalize (bool): Whether to normalize the output (prevent clipping)
            
        Returns:
            np.array: Mixed stereo audio (shape: (2, n_samples))
        """
        if not self.tracks:
            print("⚠️  No tracks to mix!")
            return None
        
        # Find the longest track to determine the mix length
        max_length = max(len(audio) for audio in self.tracks.values())
        
        # Initialize stereo mix (2 channels, max_length samples)
        mixed = np.zeros((2, max_length))
        
        # Mix each track
        for name, audio in self.tracks.items():
            # Step 1: Apply gain (volume control)
            gained = self.apply_gain(audio, self.track_gains[name])
            
            # Step 2: Apply panning (convert to stereo and position)
            stereo = self.apply_pan(gained, self.track_pans[name])
            
            # Step 3: Pad shorter tracks with silence
            if len(audio) < max_length:
                stereo = np.pad(stereo, ((0, 0), (0, max_length - len(audio))), mode='constant')
            
            # Step 4: Add this track to the mix
            mixed += stereo
        
        # Step 5: Normalize if requested to prevent clipping
        if normalize:
            mixed = mixed / np.max(np.abs(mixed))
        
        print(f"✓ Mixed {len(self.tracks)} tracks into stereo")
        
        return mixed
    
    def visualize(self):
        """
        Visualize all tracks.
        
        This creates a plot showing all the tracks in the mixer, making it
        easy to see their waveforms and settings at a glance.
        """
        n_tracks = len(self.tracks)
        
        if n_tracks == 0:
            print("⚠️  No tracks to visualize!")
            return
        
        # Create subplots for each track
        fig, axes = plt.subplots(n_tracks, 1, figsize=(14, 2 * n_tracks))
        
        # Handle single track case
        if n_tracks == 1:
            axes = [axes]
        
        # Plot each track
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
    
    A crossfade is a smooth transition between two audio tracks where one
    fades out while the other fades in. It's commonly used in DJ mixing
    and audio editing to create seamless transitions between songs.
    
    What it does:
    1. Takes the end of the first audio and fades it out
    2. Takes the beginning of the second audio and fades it in
    3. Overlaps these faded sections and mixes them together
    4. Combines everything into one continuous audio stream
    
    Args:
        audio1 (np.array): First audio signal (the one that fades out)
        audio2 (np.array): Second audio signal (the one that fades in)
        crossfade_duration (float): Crossfade duration in seconds
        sr (int): Sample rate
        
    Returns:
        np.array: Crossfaded audio (audio1 fading into audio2)
        
    Example:
        # Create a 3-second crossfade between two songs
        result = crossfade(song1, song2, crossfade_duration=3.0)
    """
    # Calculate how many samples the crossfade should be
    crossfade_samples = int(crossfade_duration * sr)
    
    # Ensure audio2 is at least as long as the crossfade
    if len(audio2) < crossfade_samples:
        audio2 = np.pad(audio2, (0, crossfade_samples - len(audio2)), mode='constant')
    
    # Handle case where audio1 is too short for crossfade
    if len(audio1) < crossfade_samples:
        # Too short, just concatenate without crossfade
        return np.concatenate([audio1, audio2])
    
    # Create fade curves
    # fade_out: goes from 1 to 0 (audio1 gets quieter)
    fade_out = np.linspace(1, 0, crossfade_samples)
    # fade_in: goes from 0 to 1 (audio2 gets louder)
    fade_in = np.linspace(0, 1, crossfade_samples)
    
    # Split audio1 into parts
    audio1_pre = audio1[:-crossfade_samples]      # Part before crossfade
    audio1_fade = audio1[-crossfade_samples:]     # Part that will fade out
    
    # Split audio2 into parts
    audio2_fade = audio2[:crossfade_samples]      # Part that will fade in
    audio2_post = audio2[crossfade_samples:]      # Part after crossfade
    
    # Apply fade curves to the crossfade sections
    audio1_faded = audio1_fade * fade_out  # Fade out audio1
    audio2_faded = audio2_fade * fade_in   # Fade in audio2
    
    # Mix the crossfade region (overlap the faded sections)
    crossfaded_region = audio1_faded + audio2_faded
    
    # Concatenate all parts: audio1_pre + crossfaded_region + audio2_post
    result = np.concatenate([audio1_pre, crossfaded_region, audio2_post])
    
    return result


def auto_duck(main_audio, ducking_audio, threshold_db=-20, ratio=3, sr=44100):
    """
    Auto-ducking: reduce main_audio volume when ducking_audio is present.
    (e.g., lower music when voice is present)
    
    Auto-ducking is a technique used in audio production where one audio track
    automatically reduces the volume of another track when it's playing.
    It's commonly used to make sure voice-over is clearly heard over background music.
    
    What it does:
    1. Monitors the ducking_audio (like voice) for volume levels
    2. When the ducking_audio gets loud enough, it reduces the main_audio volume
    3. When the ducking_audio gets quiet, it restores the main_audio volume
    4. This creates a smooth, automatic volume adjustment
    
    Args:
        main_audio (np.array): Main audio (to be ducked, like background music)
        ducking_audio (np.array): Control audio (triggers ducking, like voice)
        threshold_db (float): Threshold in dB (when ducking starts)
        ratio (float): Ducking ratio (how much to reduce volume)
        sr (int): Sample rate
        
    Returns:
        np.array: Ducked audio (main_audio with automatic volume adjustments)
        
    Example:
        # Duck background music when voice is present
        ducked_music = auto_duck(background_music, voice, threshold_db=-25, ratio=4)
    """
    # Step 1: Make sure both audios are the same length
    max_len = max(len(main_audio), len(ducking_audio))
    main_audio = np.pad(main_audio, (0, max_len - len(main_audio)), mode='constant')
    ducking_audio = np.pad(ducking_audio, (0, max_len - len(ducking_audio)), mode='constant')
    
    # Step 2: Calculate envelope of ducking signal
    # The envelope shows how loud the ducking audio is over time
    envelope = np.abs(ducking_audio)
    
    # Step 3: Smooth the envelope to avoid rapid volume changes
    from scipy.ndimage import uniform_filter1d
    window_size = int(0.01 * sr)  # 10ms window for smoothing
    envelope_smooth = uniform_filter1d(envelope, size=window_size)
    
    # Step 4: Convert to dB for easier threshold comparison
    envelope_db = 20 * np.log10(envelope_smooth + 1e-10)
    
    # Step 5: Calculate ducking gain
    # When ducking_audio is above threshold, reduce main_audio volume
    threshold = threshold_db
    gain_reduction_db = np.where(
        envelope_db > threshold,                    # If ducking_audio is loud enough
        (envelope_db - threshold) / ratio,         # Calculate how much to reduce volume
        0                                          # Otherwise, no reduction
    )
    
    # Step 6: Convert dB reduction to linear gain
    ducking_gain = 10 ** (-gain_reduction_db / 20)
    
    # Step 7: Apply ducking to main audio
    ducked = main_audio * ducking_gain
    
    return ducked


def create_stereo_from_mono(audio, width=1.0, delay_ms=10):
    """
    Create stereo from mono using Haas effect.
    
    The Haas effect (also called precedence effect) is a psychoacoustic phenomenon
    where the human brain perceives the direction of sound based on the first
    arrival time of the sound to each ear. This function uses this effect to
    create a stereo image from mono audio.
    
    What it does:
    1. Takes mono audio and creates two channels
    2. Adds a small delay to one channel (usually the right)
    3. This creates a stereo effect that makes the audio sound wider
    4. The width parameter controls how wide the stereo image is
    
    Args:
        audio (np.array): Mono audio data
        width (float): Stereo width (0-1, where 1 is maximum width)
        delay_ms (float): Delay between channels in milliseconds (usually 10-30ms)
        
    Returns:
        np.array: Stereo audio (shape: (2, n_samples))
        
    Example:
        # Create stereo from mono with 20ms delay and 80% width
        stereo = create_stereo_from_mono(mono_audio, width=0.8, delay_ms=20)
    """
    # Step 1: Calculate delay in samples
    delay_samples = int(delay_ms * 44100 / 1000)
    
    # Step 2: Create left and right channels
    left = audio  # Left channel is the original audio
    # Right channel is delayed by the specified amount
    right = np.pad(audio, (delay_samples, 0), mode='constant')[:-delay_samples]
    
    # Step 3: Apply width control using mid-side processing
    # Mid signal: what's common to both channels (center)
    mid = (left + right) / 2
    # Side signal: what's different between channels (stereo information)
    side = (left - right) / 2
    
    # Step 4: Adjust the stereo width
    # More width = more side signal, less width = more mid signal
    left_out = mid + side * width
    right_out = mid - side * width
    
    return np.array([left_out, right_out])


def showcase_mixing(audio_path1, audio_path2=None, output_dir="output", save=True):
    """
    Showcase mixing capabilities.
    
    This function demonstrates all the different mixing techniques available
    in this module. It's perfect for learning how different mixing methods
    affect the final sound.
    
    What it demonstrates:
    1. Multi-track mixing with panning and gain control
    2. Crossfading between two audio tracks
    3. Auto-ducking (automatic volume adjustment)
    4. Creating stereo from mono using Haas effect
    5. Visualizations of all the mixing results
    
    Args:
        audio_path1 (str): Path to first audio file
        audio_path2 (str): Path to second audio file (optional - if not provided, creates one by pitch shifting)
        output_dir (str): Directory to save output files
        save (bool): Whether to save the generated audio files
        
    Returns:
        dict: Dictionary containing all the mixed results
    """
    print("🎚️  Mixing Showcase")
    print("=" * 50)
    
    results = {}  # Dictionary to store all the mixing results
    
    # Step 1: Load audio files
    print("\n📂  Loading audio files...")
    y1, sr = librosa.load(audio_path1, sr=44100, mono=True)
    print(f"   ✓ Loaded {audio_path1}")
    
    if audio_path2:
        # Load the second audio file if provided
        y2, _ = librosa.load(audio_path2, sr=44100, mono=True)
        print(f"   ✓ Loaded {audio_path2}")
    else:
        # Create a second track by pitch shifting the first one
        y2 = librosa.effects.pitch_shift(y1, sr=sr, n_steps=7)
        print(f"   ✓ Created second track (pitch shifted)")
    
    # Ensure both tracks are the same length
    min_len = min(len(y1), len(y2))
    y1 = y1[:min_len]
    y2 = y2[:min_len]
    
    # 1. Multi-track mixing demonstration
    print("\n🎛️  Multi-track mixing demo...")
    mixer = Mixer(sr)
    # Add first track panned left with normal gain
    mixer.add_track("Track 1", y1, gain_db=0, pan=-0.5)
    # Add second track panned right with reduced gain
    mixer.add_track("Track 2", y2, gain_db=-3, pan=0.5)
    
    # Mix the tracks together
    mixed_stereo = mixer.mix(normalize=True)
    results['multitrack_mix'] = mixed_stereo
    
    if save:
        # Save as stereo (transpose to get correct format)
        sf.write(f"{output_dir}/mix_multitrack.wav", mixed_stereo.T, sr)
        print(f"   ✓ Multi-track mix saved")
    
    # 2. Crossfade demonstration
    print("\n🔀  Crossfade demo...")
    crossfaded = crossfade(y1, y2, crossfade_duration=2.0, sr=sr)
    results['crossfade'] = crossfaded
    
    if save:
        sf.write(f"{output_dir}/mix_crossfade.wav", crossfaded, sr)
        print(f"   ✓ Crossfade saved")
    
    # 3. Auto-ducking demonstration
    print("\n🎤  Auto-ducking demo...")
    # Duck y1 when y2 is present
    ducked = auto_duck(y1, y2, threshold_db=-25, ratio=4, sr=sr)
    results['auto_duck'] = ducked
    
    if save:
        sf.write(f"{output_dir}/mix_auto_duck.wav", ducked, sr)
        print(f"   ✓ Auto-ducked audio saved")
    
    # 4. Stereo from mono demonstration
    print("\n🎧  Stereo from mono demo...")
    # Create stereo from mono using Haas effect
    stereo = create_stereo_from_mono(y1, width=0.8, delay_ms=15)
    results['stereo_haas'] = stereo
    
    if save:
        sf.write(f"{output_dir}/mix_stereo_haas.wav", stereo.T, sr)
        print(f"   ✓ Stereo (Haas effect) saved")
    
    # Step 2: Create visualizations
    print("\n📊  Creating visualization...")
    plt.figure(figsize=(14, 10))
    
    # Plot original tracks
    plt.subplot(4, 1, 1)
    librosa.display.waveshow(y1, sr=sr)
    plt.title("Track 1")
    
    plt.subplot(4, 1, 2)
    librosa.display.waveshow(y2, sr=sr)
    plt.title("Track 2")
    
    # Plot mixed stereo
    plt.subplot(4, 1, 3)
    librosa.display.waveshow(mixed_stereo[0], sr=sr, label='Left', alpha=0.7)
    librosa.display.waveshow(mixed_stereo[1], sr=sr, label='Right', alpha=0.7)
    plt.title("Multi-track Mix (Stereo)")
    plt.legend()
    
    # Plot crossfade
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

