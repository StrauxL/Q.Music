#!/usr/bin/env python3
"""
Filter Intensity Control Demo
-----------------------------
Demonstrates how to control the intensity of high pass and low pass filters
using the mix parameter.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.daw_effects import apply_highpass_filter, apply_lowpass_filter
from scripts import config


def demo_filter_intensity():
    """Demonstrate different filter intensity levels."""
    
    print("🎛️  Filter Intensity Control Demo")
    print("=" * 50)
    
    # Load audio
    print("📁  Loading audio...")
    import librosa
    y, sr = librosa.load(config.AUDIO_PATH, mono=True, duration=120.0)  # Load first 5 seconds
    
    # Test different intensity levels
    intensity_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    
    print("\n🔊  Testing High Pass Filter Intensity...")
    highpass_results = {}
    for intensity in intensity_levels:
        print(f"   Mix: {intensity:.2f}")
        highpass_results[intensity] = apply_highpass_filter(
            y, sr, cutoff_freq=500, order=4, mix=intensity
        )
    
    print("\n🔇  Testing Low Pass Filter Intensity...")
    lowpass_results = {}
    for intensity in intensity_levels:
        print(f"   Mix: {intensity:.2f}")
        lowpass_results[intensity] = apply_lowpass_filter(
            y, sr, cutoff_freq=3000, order=4, mix=intensity
        )
    
    # Create visualization
    print("\n📊  Creating visualization...")
    fig, axes = plt.subplots(2, len(intensity_levels), figsize=(20, 8))
    
    # High pass filter results
    for i, intensity in enumerate(intensity_levels):
        axes[0, i].plot(highpass_results[intensity][:sr//2])  # Plot first 0.5 seconds
        axes[0, i].set_title(f'High Pass\nMix: {intensity:.2f}')
        axes[0, i].set_ylabel('Amplitude')
        axes[0, i].grid(True, alpha=0.3)
    
    # Low pass filter results
    for i, intensity in enumerate(intensity_levels):
        axes[1, i].plot(lowpass_results[intensity][:sr//2])  # Plot first 0.5 seconds
        axes[1, i].set_title(f'Low Pass\nMix: {intensity:.2f}')
        axes[1, i].set_ylabel('Amplitude')
        axes[1, i].set_xlabel('Samples')
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{config.OUTPUT_DIR}/filter_intensity_demo.png", dpi=150)
    print(f"   ✓ Visualization saved to {config.OUTPUT_DIR}/filter_intensity_demo.png")
    plt.show()
    
    # Save audio examples
    print("\n💾  Saving audio examples...")
    import soundfile as sf
    
    # Save high pass examples
    for intensity in [0.0, 0.5, 1.0]:
        output_path = f"{config.OUTPUT_DIR}/highpass_mix_{intensity:.1f}.wav"
        sf.write(output_path, highpass_results[intensity], sr)
        print(f"   ✓ High pass (mix={intensity:.1f}) -> {output_path}")
    
    # Save low pass examples
    for intensity in [0.0, 0.5, 1.0]:
        output_path = f"{config.OUTPUT_DIR}/lowpass_mix_{intensity:.1f}.wav"
        sf.write(output_path, lowpass_results[intensity], sr)
        print(f"   ✓ Low pass (mix={intensity:.1f}) -> {output_path}")
    
    print("\n✅  Filter intensity demo completed!")
    print("\n📝  Usage Examples:")
    print("   # Full intensity (100% filtered)")
    print("   filtered = apply_highpass_filter(audio, sr, cutoff_freq=500, mix=1.0)")
    print()
    print("   # Half intensity (50% filtered, 50% original)")
    print("   filtered = apply_highpass_filter(audio, sr, cutoff_freq=500, mix=0.5)")
    print()
    print("   # Subtle effect (25% filtered, 75% original)")
    print("   filtered = apply_highpass_filter(audio, sr, cutoff_freq=500, mix=0.25)")
    print()
    print("   # No effect (0% filtered, 100% original)")
    print("   filtered = apply_highpass_filter(audio, sr, cutoff_freq=500, mix=0.0)")


if __name__ == "__main__":
    demo_filter_intensity()
