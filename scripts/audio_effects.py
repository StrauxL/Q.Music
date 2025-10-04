"""
AUDIO EFFECTS MODULE
--------------------
Functions for creating audio mashups and applying various effects.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def create_audio_mashup(audio_path, output_dir="output", y=None, sr=None, save=True):
    """
    Create a mashup of audio segments with different effects.
    
    Args:
        audio_path (str): Path to the audio file (used if y and sr are None)
        output_dir (str): Directory to save output file
        y (np.array, optional): Audio data
        sr (int, optional): Sample rate
        save (bool): Whether to save the output file
        
    Returns:
        np.array: The mashup audio data
    """
    if y is None or sr is None:
        y, sr = librosa.load(audio_path)

    # Split audio into segments
    segment_length = len(y) // 5
    segments = []

    for i in range(5):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]
        segments.append(segment)

    # Apply different effects to each segment
    modified_segments = []

    # 1. Normal segment
    modified_segments.append(segments[0])

    # 2. Speed up segment
    modified_segments.append(librosa.effects.time_stretch(segments[1], rate=1.5))

    # 3. Pitch shifted segment
    modified_segments.append(librosa.effects.pitch_shift(segments[2], sr=sr, n_steps=4))

    # 4. Reversed segment
    modified_segments.append(segments[3][::-1])

    # 5. Echo effect segment
    echo_segment = np.zeros_like(segments[4])
    echo_segment[: len(segments[4])] = segments[4]
    decay = 0.5
    delay_samples = int(0.2 * sr)

    for i in range(1, 5):
        if delay_samples * i < len(echo_segment):
            echo = segments[4] * (decay**i)
            max_copy = min(len(echo), len(echo_segment) - delay_samples * i)
            echo_segment[delay_samples * i : delay_samples * i + max_copy] += echo[
                :max_copy
            ]

    echo_segment = echo_segment / np.max(np.abs(echo_segment))  # Normalize
    modified_segments.append(echo_segment)

    # Combine segments
    mashup = np.concatenate(modified_segments)

    # Normalize mashup
    mashup = mashup / np.max(np.abs(mashup))

    # Visualize mashup
    plt.figure(figsize=(14, 10))

    # Plot original waveform
    plt.subplot(2, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Audio")

    # Plot mashup waveform
    plt.subplot(2, 1, 2)
    librosa.display.waveshow(mashup, sr=sr)
    plt.title("Audio Mashup")

    plt.tight_layout()
    plt.show()

    if save:
        output_path = f"{output_dir}/audio_mashup.wav"
        sf.write(output_path, mashup, sr)
        print(f"✓ Mashup saved to {output_path}")

    return mashup

