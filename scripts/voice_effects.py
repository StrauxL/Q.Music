"""
VOICE EFFECTS MODULE
--------------------
Functions for applying voice transformation effects.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal
from IPython.display import Audio, display


def create_voice_changer(audio_path, output_dir="output", y=None, sr=None, save=True):
    """
    Apply voice changing effects to make interesting voice transformations.
    
    Args:
        audio_path (str): Path to the audio file (used if y and sr are None)
        output_dir (str): Directory to save output files
        y (np.array, optional): Audio data
        sr (int, optional): Sample rate
        save (bool): Whether to save the output files
        
    Returns:
        dict: Dictionary of voice effects
    """
    if y is None or sr is None:
        y, sr = librosa.load(audio_path)

    voice_effects = {}

    # 1. Chipmunk voice (high pitch)
    def chipmunk(y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=7)

    voice_effects["Chipmunk"] = chipmunk(y, sr)

    # 2. Deep voice (low pitch)
    def deep_voice(y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)

    voice_effects["Deep Voice"] = deep_voice(y, sr)

    # 3. Robot voice (vocoder-like)
    def robot_voice(y, sr):
        # Simple vocoder-like effect
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Create a simple "robotic" carrier signal
        hopped_y = librosa.util.frame(y, frame_length=2048, hop_length=512)
        robot = np.zeros_like(y)

        # Create a modulated sine wave based on pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0 = np.nan_to_num(f0)
        t = np.arange(len(y)) / sr

        carrier = np.sin(2 * np.pi * 200 * t)  # Base carrier frequency

        # Modulate with original audio
        robot = carrier * y

        # Normalize
        return robot / np.max(np.abs(robot))

    voice_effects["Robot"] = robot_voice(y, sr)

    # 4. Helium voice (very high pitch + faster)
    def helium_voice(y, sr):
        # Speed up and raise pitch
        stretched = librosa.effects.time_stretch(y, rate=1.25)
        return librosa.effects.pitch_shift(stretched, sr=sr, n_steps=12)

    voice_effects["Helium"] = helium_voice(y, sr)

    # 5. Underwater effect
    def underwater(y, sr):
        # Apply lowpass filter
        b, a = signal.butter(4, 0.2, "lowpass")
        filtered = signal.filtfilt(b, a, y)

        # Add reverb (simple implementation)
        reverb = np.zeros_like(filtered)
        delay_samples = int(0.05 * sr)  # 50ms delay
        reverb[delay_samples:] = filtered[:-delay_samples] * 0.5

        result = filtered + reverb
        return result / np.max(np.abs(result))

    voice_effects["Underwater"] = underwater(y, sr)

    # Visualize effects
    plt.figure(figsize=(15, 12))

    # Plot original waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Voice")

    # Plot each effect
    for i, (name, effect) in enumerate(voice_effects.items(), 2):
        plt.subplot(3, 2, i)
        librosa.display.waveshow(effect, sr=sr)
        plt.title(name)

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
    
    Args:
        audio_path (str): Path to the audio file
        output_dir (str): Directory to save output file
        save (bool): Whether to save the output file
        
    Returns:
        tuple: The nightcore audio data and sample rate
    """
    y, sr = librosa.load(audio_path)
    segment_length = len(y) // 5
    segment = y[
        2 * segment_length + (segment_length // 4) : 2 * segment_length
        + segment_length
        + (segment_length // 2)
    ]
    
    # Apply time stretch and pitch shift
    fast_segment = librosa.effects.time_stretch(segment, rate=1.25)
    nightcore_segment = librosa.effects.pitch_shift(fast_segment, sr=sr, n_steps=2)
    # Note: n_steps = -3 for male version
    
    # Apply echo effect
    echo_segment = (
        conv := np.convolve(
            nightcore_segment,
            np.bincount(
                np.arange(5) * int(0.2 * sr),
                weights=0.5 ** np.arange(5),
                minlength=int(0.2 * sr) * 4 + 1,
            ),
        )[: len(nightcore_segment)]
    ) / np.max(np.abs(conv))
    
    # Normalize
    normalized_segment = echo_segment / np.max(np.abs(echo_segment))
    
    # Display audio in IPython environment
    display(Audio(normalized_segment, rate=sr, autoplay=True))
    
    # Visualize
    librosa.display.waveshow(normalized_segment, sr=sr)
    plt.title("Nightcore Effect")
    plt.show()
    
    if save:
        output_path = f"{output_dir}/nightcore_effect.wav"
        sf.write(output_path, normalized_segment, sr)
        print(f"✓ Nightcore effect saved to {output_path}")
    
    return normalized_segment, sr

