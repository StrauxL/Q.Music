"""
AUDIO INFO MODULE
-----------------
Functions for analyzing and displaying audio file information.
"""

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def basic_audio_info(audio_path):
    """
    Display basic information about the audio file and visualize waveform and spectrogram.
    
    Args:
        audio_path (str): Path to the audio file
        
    Returns:
        tuple: Audio data (y) and sample rate (sr)
    """
    try:
        # Load audio file
        y, sr = librosa.load(audio_path)

        # Print basic info
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Number of samples: {len(y)}")

        # Create figure for visualizations
        plt.figure(figsize=(14, 10))

        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Plot spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")

        plt.tight_layout()
        plt.show()

        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None

