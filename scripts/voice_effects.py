"""
VOICE EFFECTS MODULE
--------------------
Functions for applying voice transformation effects.

This module provides beginner-friendly functions to transform voices and create
fun voice effects. It's perfect for learning how to manipulate audio to create
different character voices or special effects.

Key Concepts for Beginners:
- Voice Transformation: Changing how a voice sounds
- Pitch Shifting: Making voices higher or lower
- Time Stretching: Making voices faster or slower
- Filtering: Removing certain frequencies to create effects
- Echo/Reverb: Adding space and depth to voices

Author: Q.Wave Team
"""

# Import necessary libraries for audio processing and visualization
import librosa             # Main audio processing library
import librosa.display     # For audio visualization
import matplotlib.pyplot as plt  # For creating plots and graphs
import numpy as np         # For numerical operations and arrays
import soundfile as sf     # For reading and writing audio files
from scipy import signal   # For signal processing functions
from IPython.display import Audio, display  # For playing audio in Jupyter notebooks


def create_voice_changer(audio_path, output_dir="output", y=None, sr=None, save=True):
    """
    Apply voice changing effects to make interesting voice transformations.
    
    This function takes an audio file and creates 5 different voice effects:
    1. Chipmunk voice (high pitch)
    2. Deep voice (low pitch)
    3. Robot voice (vocoder-like effect)
    4. Helium voice (very high pitch + faster)
    5. Underwater effect (lowpass filter + reverb)
    
    What it does:
    1. Loads the audio file
    2. Applies different voice transformation effects
    3. Creates visualizations showing the original vs transformed audio
    4. Saves each effect as a separate audio file
    5. Returns a dictionary with all the transformed voices
    
    Args:
        audio_path (str): Path to the audio file (used if y and sr are None)
        output_dir (str): Directory to save output files
        y (np.array, optional): Audio data (if you already have it loaded)
        sr (int, optional): Sample rate (if you already have it loaded)
        save (bool): Whether to save the output files
        
    Returns:
        dict: Dictionary of voice effects containing:
            - "Chipmunk": High-pitched voice
            - "Deep Voice": Low-pitched voice
            - "Robot": Robotic-sounding voice
            - "Helium": Very high-pitched and fast voice
            - "Underwater": Filtered voice with reverb
    """
    # Step 1: Load audio if not already provided
    if y is None or sr is None:
        y, sr = librosa.load(audio_path)

    voice_effects = {}  # Dictionary to store all the voice effects

    # Effect 1: Chipmunk voice (high pitch)
    # This makes the voice sound like a chipmunk by raising the pitch
    def chipmunk(y, sr):
        # n_steps=7 means 7 semitones higher (like going from C to G on a piano)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=7)

    voice_effects["Chipmunk"] = chipmunk(y, sr)

    # Effect 2: Deep voice (low pitch)
    # This makes the voice sound deeper by lowering the pitch
    def deep_voice(y, sr):
        # n_steps=-6 means 6 semitones lower (like going from C to F# on a piano)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)

    voice_effects["Deep Voice"] = deep_voice(y, sr)

    # Effect 3: Robot voice (vocoder-like)
    # This creates a robotic effect by modulating the audio with a sine wave
    def robot_voice(y, sr):
        # Extract pitch information from the audio
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Create a simple "robotic" carrier signal
        hopped_y = librosa.util.frame(y, frame_length=2048, hop_length=512)
        robot = np.zeros_like(y)

        # Create a modulated sine wave based on pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0 = np.nan_to_num(f0)  # Replace NaN values with 0
        t = np.arange(len(y)) / sr  # Time array

        carrier = np.sin(2 * np.pi * 200 * t)  # Base carrier frequency (200 Hz)

        # Modulate with original audio to create robotic effect
        robot = carrier * y

        # Normalize to prevent clipping
        return robot / np.max(np.abs(robot))

    voice_effects["Robot"] = robot_voice(y, sr)

    # Effect 4: Helium voice (very high pitch + faster)
    # This combines time stretching and pitch shifting for a helium effect
    def helium_voice(y, sr):
        # First, speed up the audio (1.25x faster)
        stretched = librosa.effects.time_stretch(y, rate=1.25)
        # Then, raise the pitch significantly (12 semitones higher)
        return librosa.effects.pitch_shift(stretched, sr=sr, n_steps=12)

    voice_effects["Helium"] = helium_voice(y, sr)

    # Effect 5: Underwater effect
    # This creates an underwater sound by filtering and adding reverb
    def underwater(y, sr):
        # Apply lowpass filter to remove high frequencies
        # This makes it sound like it's coming through water
        b, a = signal.butter(4, 0.2, "lowpass")  # 4th order Butterworth filter
        filtered = signal.filtfilt(b, a, y)  # Apply filter in both directions

        # Add simple reverb effect
        reverb = np.zeros_like(filtered)
        delay_samples = int(0.05 * sr)  # 50ms delay
        reverb[delay_samples:] = filtered[:-delay_samples] * 0.5  # Add delayed copy

        result = filtered + reverb
        return result / np.max(np.abs(result))  # Normalize

    voice_effects["Underwater"] = underwater(y, sr)

    # Step 2: Visualize all the effects
    # Create a plot showing the original voice and all the transformed versions
    plt.figure(figsize=(15, 12))

    # Plot original waveform
    plt.subplot(3, 2, 1)  # 3 rows, 2 columns, position 1
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Voice")

    # Plot each effect
    for i, (name, effect) in enumerate(voice_effects.items(), 2):
        plt.subplot(3, 2, i)  # 3 rows, 2 columns, position i
        librosa.display.waveshow(effect, sr=sr)
        plt.title(name)

        # Save each effect if requested
        if save:
            output_path = f"{output_dir}/voice_{name.lower().replace(' ', '_')}.wav"
            sf.write(output_path, effect, sr)
            print(f"✓ Voice effect saved to {output_path}")

    plt.tight_layout()
    plt.show()

    return voice_effects


def create_nightcore_effect(audio_path, output_dir="output", save=True):
    """
    Create a nightcore effect (faster tempo + higher pitch with optional echo).
    
    Nightcore is a music style that involves speeding up songs and raising their pitch.
    This function creates that effect by combining time stretching and pitch shifting.
    
    What it does:
    1. Loads the audio file
    2. Extracts a specific segment from the middle of the song
    3. Speeds up the audio (1.25x faster)
    4. Raises the pitch (2 semitones higher)
    5. Adds an echo effect for extra depth
    6. Normalizes the audio to prevent clipping
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save output file
        save (bool): Whether to save the output file
        
    Returns:
        tuple: The nightcore audio data and sample rate
            - normalized_segment: The processed audio data
            - sr: Sample rate of the audio
    """
    # Step 1: Load the audio file
    y, sr = librosa.load(audio_path)
    
    # Step 2: Extract a specific segment from the middle of the song
    # This creates a more interesting nightcore effect by using a specific part
    segment_length = len(y) // 5  # Divide the song into 5 parts
    segment = y[
        2 * segment_length + (segment_length // 4) : 2 * segment_length
        + segment_length
        + (segment_length // 2)
    ]  # Take a segment from the middle part
    
    # Step 3: Apply time stretch and pitch shift
    # First, speed up the audio (1.25x faster)
    fast_segment = librosa.effects.time_stretch(segment, rate=1.25)
    
    # Then, raise the pitch (2 semitones higher)
    nightcore_segment = librosa.effects.pitch_shift(fast_segment, sr=sr, n_steps=2)
    # Note: n_steps = -3 for male version (to make it sound more natural)
    
    # Step 4: Apply echo effect
    # This creates multiple delayed copies of the audio with decreasing volume
    echo_segment = (
        conv := np.convolve(  # Convolution creates the echo effect
            nightcore_segment,
            np.bincount(  # Create a pattern for the echoes
                np.arange(5) * int(0.2 * sr),  # Echo delays: 0, 0.2s, 0.4s, 0.6s, 0.8s
                weights=0.5 ** np.arange(5),   # Echo volumes: 1.0, 0.5, 0.25, 0.125, 0.0625
                minlength=int(0.2 * sr) * 4 + 1,  # Minimum length for the pattern
            ),
        )[: len(nightcore_segment)]  # Trim to original length
    ) / np.max(np.abs(conv))  # Normalize to prevent clipping
    
    # Step 5: Final normalization
    # Ensure the audio doesn't get too loud and cause distortion
    normalized_segment = echo_segment / np.max(np.abs(echo_segment))
    
    # Step 6: Display audio in IPython environment (if running in Jupyter)
    # This allows you to play the audio directly in the notebook
    display(Audio(normalized_segment, rate=sr, autoplay=True))
    
    # Step 7: Visualize the result
    # Create a plot showing the nightcore effect
    librosa.display.waveshow(normalized_segment, sr=sr)
    plt.title("Nightcore Effect")
    plt.show()
    
    # Step 8: Save the result if requested
    if save:
        output_path = f"{output_dir}/nightcore_effect.wav"
        sf.write(output_path, normalized_segment, sr)
        print(f"✓ Nightcore effect saved to {output_path}")
    
    return normalized_segment, sr

