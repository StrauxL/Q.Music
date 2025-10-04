"""
EXAMPLE USAGE - DAW Features
-----------------------------
Quick examples showing how to use the new DAW features.
Run individual functions or uncomment sections to test.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import config
from scripts.daw_effects import *
from scripts.synthesis import SimpleSynth, Oscillator, ADSR
from scripts.mixing import Mixer, crossfade, auto_duck
from scripts.analysis import beat_detector, loudness_meter, spectral_features
import librosa
import soundfile as sf
import numpy as np


def example_compression():
    """Example: Apply compression to audio."""
    print("\n📊 Example: Compression")
    print("-" * 50)
    
    # Load audio
    y, sr = librosa.load(config.AUDIO_PATH, duration=10)
    
    # Apply compression
    compressed = apply_compressor(
        y, sr,
        threshold_db=-20,  # Compress when above -20 dB
        ratio=4,           # 4:1 compression ratio
        attack_time=0.005, # Fast attack
        release_time=0.1   # Medium release
    )
    
    # Save
    sf.write(f"{config.OUTPUT_DIR}/example_compressed.wav", compressed, sr)
    print("✓ Compressed audio saved!")


def example_eq():
    """Example: Apply EQ to boost presence."""
    print("\n🎛️ Example: Parametric EQ")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=10)
    
    # High-pass filter (remove low rumble)
    y = apply_parametric_eq(y, sr, freq=80, filter_type='highpass')
    
    # Boost presence at 3kHz
    y = apply_parametric_eq(y, sr, freq=3000, gain_db=5, q=1.5, filter_type='peak')
    
    sf.write(f"{config.OUTPUT_DIR}/example_eq.wav", y, sr)
    print("✓ EQ'd audio saved!")


def example_reverb_delay():
    """Example: Add reverb and delay."""
    print("\n🌊 Example: Reverb + Delay")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=10)
    
    # Add reverb
    y = apply_reverb(y, sr, room_size=0.6, damping=0.5, wet_level=0.3)
    
    # Add delay
    y = apply_delay(y, sr, delay_time=0.375, feedback=0.4, mix=0.4)
    
    sf.write(f"{config.OUTPUT_DIR}/example_reverb_delay.wav", y, sr)
    print("✓ Reverb + Delay audio saved!")


def example_synthesis():
    """Example: Create music with synthesizer."""
    print("\n🎹 Example: Synthesis")
    print("-" * 50)
    
    synth = SimpleSynth(sample_rate=44100)
    
    # Create a chord progression
    chords = [
        ['C4', 'E4', 'G4'],   # C major
        ['A3', 'C4', 'E4'],   # A minor
        ['F3', 'A3', 'C4'],   # F major
        ['G3', 'B3', 'D4'],   # G major
    ]
    
    progression = []
    for chord_notes in chords:
        chord = synth.play_chord(
            chord_notes,
            duration=2.0,
            waveform='sawtooth',
            attack=0.1,
            decay=0.2,
            sustain=0.7,
            release=0.5
        )
        progression.append(chord)
    
    # Concatenate all chords
    music = np.concatenate(progression)
    
    sf.write(f"{config.OUTPUT_DIR}/example_synth_chords.wav", music, synth.sr)
    print("✓ Synthesized chord progression saved!")


def example_bass_melody():
    """Example: Create a bass line and melody."""
    print("\n🎵 Example: Bass + Melody")
    print("-" * 50)
    
    synth = SimpleSynth(sample_rate=44100)
    
    # Bass line (low notes, short attack)
    bass_notes = ['C2', 'C2', 'G2', 'C2', 'A2', 'A2', 'F2', 'G2']
    bass = synth.play_sequence(
        bass_notes,
        note_duration=0.5,
        waveform='sawtooth',
        attack=0.01,
        decay=0.1,
        sustain=0.3,
        release=0.1
    )
    
    # Melody (higher notes, longer sustain)
    melody_notes = ['C5', 'E5', 'G5', 'E5', 'A5', 'G5', 'F5', 'E5']
    melody = synth.play_sequence(
        melody_notes,
        note_duration=0.5,
        waveform='square',
        attack=0.05,
        decay=0.1,
        sustain=0.6,
        release=0.2
    )
    
    # Mix them together
    mixer = Mixer(sample_rate=synth.sr)
    mixer.add_track("Bass", bass, gain_db=0, pan=-0.2)
    mixer.add_track("Melody", melody, gain_db=-3, pan=0.2)
    mixed = mixer.mix(normalize=True)
    
    # Convert stereo to mono for saving (or save stereo)
    sf.write(f"{config.OUTPUT_DIR}/example_bass_melody.wav", mixed.T, synth.sr)
    print("✓ Bass + Melody saved!")


def example_mixing():
    """Example: Multi-track mixing."""
    print("\n🎛️ Example: Multi-track Mixing")
    print("-" * 50)
    
    # Load two audio files (or create variations)
    y1, sr = librosa.load(config.AUDIO_PATH, duration=10)
    y2 = librosa.effects.pitch_shift(y1, sr=sr, n_steps=7)  # Pitch shift for second track
    
    # Create mixer
    mixer = Mixer(sample_rate=sr)
    mixer.add_track("Track 1", y1, gain_db=0, pan=-0.5)    # Left
    mixer.add_track("Track 2", y2, gain_db=-3, pan=0.5)    # Right
    
    # Mix
    mixed = mixer.mix(normalize=True)
    
    sf.write(f"{config.OUTPUT_DIR}/example_multitrack_mix.wav", mixed.T, sr)
    print("✓ Multi-track mix saved!")


def example_beat_detection():
    """Example: Detect beats and tempo."""
    print("\n🥁 Example: Beat Detection")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=30)
    
    # Detect beats
    beat_info = beat_detector(y, sr)
    
    print(f"Detected tempo: {beat_info['tempo']:.1f} BPM")
    print(f"Number of beats: {len(beat_info['beat_times'])}")
    print(f"Beat times: {beat_info['beat_times'][:10]}...")  # First 10 beats
    
    return beat_info


def example_loudness_analysis():
    """Example: Analyze loudness."""
    print("\n🔊 Example: Loudness Analysis")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=30)
    
    # RMS loudness
    rms = loudness_meter(y, sr, meter_type='rms')
    print(f"RMS Mean: {rms['rms_mean']:.2f} dB")
    print(f"RMS Max: {rms['rms_max']:.2f} dB")
    
    # Peak loudness
    peak = loudness_meter(y, sr, meter_type='peak')
    print(f"Peak Mean: {peak['peak_mean']:.2f} dB")
    print(f"Peak Max: {peak['peak_max']:.2f} dB")


def example_mastering_chain():
    """Example: Complete mastering chain."""
    print("\n✨ Example: Mastering Chain")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=10)
    
    # Step 1: High-pass filter (remove sub-bass rumble)
    print("  1. High-pass filter...")
    y = apply_parametric_eq(y, sr, freq=40, filter_type='highpass')
    
    # Step 2: Compression
    print("  2. Compression...")
    y = apply_compressor(y, sr, threshold_db=-18, ratio=3, 
                        attack_time=0.01, release_time=0.1)
    
    # Step 3: EQ boost (add presence)
    print("  3. Presence boost...")
    y = apply_parametric_eq(y, sr, freq=3000, gain_db=3, q=1.5, filter_type='peak')
    
    # Step 4: Limiter (prevent clipping)
    print("  4. Limiting...")
    y = apply_limiter(y, threshold_db=-1)
    
    sf.write(f"{config.OUTPUT_DIR}/example_mastered.wav", y, sr)
    print("✓ Mastered audio saved!")


def example_vocal_processing():
    """Example: Vocal processing chain."""
    print("\n🎤 Example: Vocal Processing")
    print("-" * 50)
    
    y, sr = librosa.load(config.AUDIO_PATH, duration=10)
    
    # Step 1: Gate (remove background noise)
    print("  1. Noise gate...")
    y = apply_gate(y, sr, threshold_db=-45, attack_time=0.001, release_time=0.1)
    
    # Step 2: High-pass filter (remove low-end rumble)
    print("  2. High-pass filter...")
    y = apply_parametric_eq(y, sr, freq=80, filter_type='highpass')
    
    # Step 3: De-esser (reduce harsh sibilance at 6-8kHz)
    print("  3. De-essing...")
    y = apply_parametric_eq(y, sr, freq=7000, gain_db=-4, q=2, filter_type='peak')
    
    # Step 4: Compression (even out dynamics)
    print("  4. Compression...")
    y = apply_compressor(y, sr, threshold_db=-18, ratio=4, 
                        attack_time=0.005, release_time=0.1)
    
    # Step 5: Presence boost
    print("  5. Presence boost...")
    y = apply_parametric_eq(y, sr, freq=3000, gain_db=4, q=1.2, filter_type='peak')
    
    # Step 6: Reverb (add space)
    print("  6. Reverb...")
    y = apply_reverb(y, sr, room_size=0.4, damping=0.6, wet_level=0.25)
    
    sf.write(f"{config.OUTPUT_DIR}/example_vocal_processed.wav", y, sr)
    print("✓ Processed vocal saved!")


def main():
    """Run example demonstrations."""
    print("\n" + "=" * 70)
    print(" 📚  DAW FEATURES - EXAMPLE USAGE")
    print("=" * 70)
    
    print("\nSelect an example to run:")
    print("1.  Compression")
    print("2.  Parametric EQ")
    print("3.  Reverb + Delay")
    print("4.  Synthesis (Chords)")
    print("5.  Bass + Melody")
    print("6.  Multi-track Mixing")
    print("7.  Beat Detection")
    print("8.  Loudness Analysis")
    print("9.  Mastering Chain")
    print("10. Vocal Processing")
    print("11. Run All Examples")
    print("0.  Exit")
    
    choice = input("\n👉  Select option (0-11): ").strip()
    
    # Check if audio file exists
    if not os.path.exists(config.AUDIO_PATH) and choice not in ['4', '5', '0']:
        print(f"\n⚠️  Warning: {config.AUDIO_PATH} not found!")
        print("Some examples require an audio file. Update config.AUDIO_PATH or use synthesis examples (4, 5).")
        return
    
    examples = {
        '1': example_compression,
        '2': example_eq,
        '3': example_reverb_delay,
        '4': example_synthesis,
        '5': example_bass_melody,
        '6': example_mixing,
        '7': example_beat_detection,
        '8': example_loudness_analysis,
        '9': example_mastering_chain,
        '10': example_vocal_processing,
    }
    
    if choice == '11':
        print("\n🚀 Running all examples...\n")
        for func in examples.values():
            try:
                func()
            except Exception as e:
                print(f"   ⚠️  Error: {e}")
        print("\n✅ All examples completed!")
    elif choice in examples:
        try:
            examples[choice]()
            print("\n✅ Example completed!")
        except Exception as e:
            print(f"\n❌ Error: {e}")
    elif choice == '0':
        print("\n👋 Goodbye!")
    else:
        print("\n❌ Invalid choice!")
    
    print(f"\n📁 Check {config.OUTPUT_DIR}/ for output files\n")


if __name__ == "__main__":
    main()

