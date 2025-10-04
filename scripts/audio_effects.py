"""
AUDIO EFFECTS MODULE
--------------------
Functions for creating audio mashups and applying various effects.

This module provides beginner-friendly functions to create audio mashups by
combining different audio effects. It's perfect for learning how to manipulate
audio and create interesting sound combinations.

Key Concepts for Beginners:
- Audio Mashup: Combining different audio segments with various effects
- Time Stretching: Making audio faster or slower without changing pitch
- Pitch Shifting: Changing the pitch (high/low) without changing speed
- Echo Effect: Adding delayed copies of the audio for a "reverb" sound
- Audio Normalization: Making sure audio doesn't get too loud and distort

Author: Q.Wave Team
"""

# Import necessary libraries for audio processing and visualization
import librosa             # Main audio processing library
import librosa.display     # For audio visualization
import matplotlib.pyplot as plt  # For creating plots and graphs
import numpy as np         # For numerical operations and arrays
import soundfile as sf     # For reading and writing audio files


def create_audio_mashup(audio_path, output_dir="output", y=None, sr=None, save=True):
    """
    Create a mashup of audio segments with different effects.
    
    This function takes an audio file and creates a creative mashup by:
    1. Splitting the audio into 5 equal segments
    2. Applying different effects to each segment
    3. Combining them back together to create a unique sound
    
    Effects applied:
    - Segment 1: Original (no changes)
    - Segment 2: Speed up (1.5x faster)
    - Segment 3: Pitch shift (higher pitch)
    - Segment 4: Reversed (played backwards)
    - Segment 5: Echo effect (with multiple delayed copies)
    
    Args:
        audio_path (str): Path to the audio file (used if y and sr are None)
        output_dir (str): Directory to save output file
        y (np.array, optional): Audio data (if you already have it loaded)
        sr (int, optional): Sample rate (if you already have it loaded)
        save (bool): Whether to save the output file
        
    Returns:
        np.array: The mashup audio data
    """
    # Step 1: Load audio if not already provided
    if y is None or sr is None:
        y, sr = librosa.load(audio_path)

    # Step 2: Split audio into 5 equal segments
    # This creates 5 different parts of the song to work with
    segment_length = len(y) // 5  # Calculate length of each segment
    segments = []  # List to store each segment

    for i in range(5):
        start = i * segment_length      # Start position of this segment
        end = start + segment_length    # End position of this segment
        segment = y[start:end]          # Extract this segment
        segments.append(segment)        # Add to our list

    # Step 3: Apply different effects to each segment
    modified_segments = []  # List to store the modified segments

    # Effect 1: Normal segment (no changes)
    # This keeps one part of the original audio unchanged
    modified_segments.append(segments[0])

    # Effect 2: Speed up segment (time stretching)
    # Makes the audio 1.5x faster without changing the pitch
    # rate=1.5 means 1.5x speed (faster)
    modified_segments.append(librosa.effects.time_stretch(segments[1], rate=1.5))

    # Effect 3: Pitch shifted segment
    # Changes the pitch (makes it higher) without changing the speed
    # n_steps=4 means 4 semitones higher (like going from C to E on a piano)
    modified_segments.append(librosa.effects.pitch_shift(segments[2], sr=sr, n_steps=4))

    # Effect 4: Reversed segment
    # Plays the audio backwards by reversing the array
    # [::-1] is Python's way to reverse an array
    modified_segments.append(segments[3][::-1])

    # Effect 5: Echo effect segment
    # Creates multiple delayed copies of the audio with decreasing volume
    echo_segment = np.zeros_like(segments[4])  # Create empty array same size as segment
    echo_segment[: len(segments[4])] = segments[4]  # Start with original audio
    
    decay = 0.5  # How much each echo gets quieter (50% quieter each time)
    delay_samples = int(0.2 * sr)  # Delay time: 0.2 seconds

    # Create 4 echoes with decreasing volume
    for i in range(1, 5):
        if delay_samples * i < len(echo_segment):
            # Create echo: original audio * decay factor
            echo = segments[4] * (decay**i)  # Each echo is quieter than the last
            
            # Calculate how much of the echo we can fit
            max_copy = min(len(echo), len(echo_segment) - delay_samples * i)
            
            # Add the echo to the segment at the right time
            echo_segment[delay_samples * i : delay_samples * i + max_copy] += echo[:max_copy]

    # Normalize the echo segment to prevent clipping
    echo_segment = echo_segment / np.max(np.abs(echo_segment))
    modified_segments.append(echo_segment)

    # Step 4: Combine all segments back together
    # Concatenate means "join end-to-end"
    mashup = np.concatenate(modified_segments)

    # Step 5: Normalize the final mashup
    # This ensures the audio doesn't get too loud and cause distortion
    mashup = mashup / np.max(np.abs(mashup))

    # Step 6: Visualize the results
    # Create a plot showing original vs mashup
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

    # Step 7: Save the mashup if requested
    if save:
        output_path = f"{output_dir}/audio_mashup.wav"
        sf.write(output_path, mashup, sr)
        print(f"✓ Mashup saved to {output_path}")

    return mashup

