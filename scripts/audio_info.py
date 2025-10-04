"""
AUDIO INFO MODULE
-----------------
Functions for analyzing and displaying audio file information.

This module provides beginner-friendly functions to get basic information about
audio files and visualize them. It's perfect for understanding what's in your
audio files before processing them.

Key Concepts for Beginners:
- Sample Rate: How many audio samples per second (usually 44100 Hz)
- Duration: How long the audio is in seconds
- Waveform: Visual representation of the audio's amplitude over time
- Spectrogram: Visual representation of frequencies over time
- Amplitude: How loud the audio is at any given moment

Author: Q.Wave Team
"""

# Import necessary libraries for audio processing and visualization
import librosa             # Main audio processing library
import librosa.display     # For audio visualization
import matplotlib.pyplot as plt  # For creating plots and graphs
import numpy as np         # For numerical operations and arrays


def basic_audio_info(audio_path):
    """
    Display basic information about the audio file and visualize waveform and spectrogram.
    
    This function is perfect for beginners who want to understand what's in their audio files.
    It loads an audio file, shows basic information about it, and creates two helpful
    visualizations: a waveform and a spectrogram.
    
    What it shows:
    1. Basic file information (sample rate, duration, number of samples)
    2. Waveform: Shows the audio's amplitude (loudness) over time
    3. Spectrogram: Shows which frequencies are present over time
    
    Args:
        audio_path (str): Path to the audio file (e.g., "song.mp3", "audio.wav")
        
    Returns:
        tuple: Audio data (y) and sample rate (sr)
            - y: The actual audio samples as a numpy array
            - sr: Sample rate (samples per second, usually 44100)
    """
    try:
        # Step 1: Load the audio file
        # librosa.load() automatically converts to mono and resamples to 22050 Hz by default
        # It returns two things: the audio data (y) and sample rate (sr)
        y, sr = librosa.load(audio_path)

        # Step 2: Calculate and display basic information
        duration = librosa.get_duration(y=y, sr=sr)  # Calculate duration in seconds
        
        print("📊 Audio File Information:")
        print("=" * 30)
        print(f"Sample rate: {sr} Hz")                    # How many samples per second
        print(f"Duration: {duration:.2f} seconds")        # How long the audio is
        print(f"Number of samples: {len(y)}")            # Total number of audio samples
        print(f"File size: {len(y) * 2 / 1024 / 1024:.2f} MB (approx)")  # Approximate file size

        # Step 3: Create visualizations
        # We'll create a figure with two plots: waveform and spectrogram
        plt.figure(figsize=(14, 10))

        # Plot 1: Waveform
        # The waveform shows the audio's amplitude (loudness) over time
        # Think of it like the "shape" of the sound wave
        plt.subplot(2, 1, 1)  # Create first subplot (2 rows, 1 column, position 1)
        librosa.display.waveshow(y, sr=sr)  # Plot the waveform
        plt.title("Waveform - Audio Amplitude Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Amplitude")
        plt.grid(True, alpha=0.3)  # Add a subtle grid for easier reading

        # Plot 2: Spectrogram
        # The spectrogram shows which frequencies are present over time
        # Colors represent intensity: brighter = louder at that frequency
        plt.subplot(2, 1, 2)  # Create second subplot (2 rows, 1 column, position 2)
        
        # Calculate the spectrogram using Short-Time Fourier Transform (STFT)
        D = librosa.stft(y)  # STFT gives us frequency content over time
        magnitude = np.abs(D)  # Get the magnitude (strength) of each frequency
        magnitude_db = librosa.amplitude_to_db(magnitude, ref=np.max)  # Convert to decibels
        
        # Display the spectrogram
        librosa.display.specshow(magnitude_db, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")  # Add color bar showing dB levels
        plt.title("Spectrogram - Frequency Content Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency (Hz)")
        
        # Make the layout look nice
        plt.tight_layout()
        plt.show()

        return y, sr
        
    except Exception as e:
        # If something goes wrong, print an error message
        print(f"❌ Error loading audio: {e}")
        print("Make sure the file path is correct and the file is a valid audio format.")
        return None, None

