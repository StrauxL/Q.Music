"""
DAW FEATURES DEMO
-----------------
Comprehensive demonstration of all DAW-like features including:
- Studio effects (compression, EQ, reverb, delay, etc.)
- Synthesis (oscillators, ADSR, synth)
- Mixing (multi-track, crossfading, auto-ducking)
- Analysis (spectrum, beat detection, metering)
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import config
from scripts.daw_effects import create_daw_effects_showcase
from scripts.synthesis import showcase_synthesis
from scripts.mixing import showcase_mixing
from scripts.analysis import showcase_analysis


def main():
    """Run all DAW feature demonstrations."""
    
    print("\n" + "=" * 70)
    print(" 🎛️  DAW FEATURES SHOWCASE - ABLETON-STYLE AUDIO PROCESSING")
    print("=" * 70 + "\n")
    
    # Check if audio file exists
    if not os.path.exists(config.AUDIO_PATH):
        print(f"⚠️  Warning: {config.AUDIO_PATH} not found!")
        print("Using default audio path from config...")
    
    print("\n📋  MENU:")
    print("1. Studio Effects (Compression, EQ, Reverb, Delay, Chorus, etc.)")
    print("2. Synthesis (Oscillators, ADSR, Synth)")
    print("3. Mixing (Multi-track, Crossfade, Auto-duck)")
    print("4. Analysis (Spectrum, Beat Detection, Metering)")
    print("5. Run All Demos")
    print("0. Exit")
    
    choice = input("\n👉  Select option (0-5): ").strip()
    
    if choice == "1":
        demo_studio_effects()
    elif choice == "2":
        demo_synthesis()
    elif choice == "3":
        demo_mixing()
    elif choice == "4":
        demo_analysis()
    elif choice == "5":
        run_all_demos()
    elif choice == "0":
        print("\n👋  Goodbye!")
        return
    else:
        print("\n❌  Invalid choice!")


def demo_studio_effects():
    """Demo studio effects."""
    print("\n" + "=" * 70)
    print(" 🎚️  STUDIO EFFECTS DEMO")
    print("=" * 70 + "\n")
    
    print("This demo will apply professional audio effects:")
    print("  • Compression (dynamic range control)")
    print("  • Parametric EQ (frequency shaping)")
    print("  • Reverb (space simulation)")
    print("  • Delay/Echo (time-based effect)")
    print("  • Chorus (modulation effect)")
    print("  • Flanger (jet plane effect)")
    print("  • Distortion (harmonic saturation)")
    print("  • Limiter (peak control)")
    print("  • Gate (noise reduction)")
    print()
    
    try:
        effects = create_daw_effects_showcase(
            config.AUDIO_PATH, 
            output_dir=config.OUTPUT_DIR, 
            save=True
        )
        print("\n✅  Studio effects demo completed!")
        print(f"📁  Check {config.OUTPUT_DIR}/ for output files")
    except Exception as e:
        print(f"\n❌  Error: {e}")


def demo_synthesis():
    """Demo synthesis capabilities."""
    print("\n" + "=" * 70)
    print(" 🎹  SYNTHESIS DEMO")
    print("=" * 70 + "\n")
    
    print("This demo will showcase synthesis capabilities:")
    print("  • Oscillators (sine, square, sawtooth, triangle)")
    print("  • ADSR envelopes (attack, decay, sustain, release)")
    print("  • LFO modulation (tremolo effect)")
    print("  • Chord synthesis")
    print("  • Melody sequencing")
    print()
    
    try:
        results = showcase_synthesis(
            output_dir=config.OUTPUT_DIR, 
            save=True
        )
        print("\n✅  Synthesis demo completed!")
        print(f"📁  Check {config.OUTPUT_DIR}/ for output files")
    except Exception as e:
        print(f"\n❌  Error: {e}")


def demo_mixing():
    """Demo mixing tools."""
    print("\n" + "=" * 70)
    print(" 🎛️  MIXING DEMO")
    print("=" * 70 + "\n")
    
    print("This demo will showcase mixing tools:")
    print("  • Multi-track mixing with gain and pan")
    print("  • Crossfading between tracks")
    print("  • Auto-ducking (sidechain-like)")
    print("  • Stereo width enhancement")
    print()
    
    try:
        results = showcase_mixing(
            config.AUDIO_PATH,
            output_dir=config.OUTPUT_DIR,
            save=True
        )
        print("\n✅  Mixing demo completed!")
        print(f"📁  Check {config.OUTPUT_DIR}/ for output files")
    except Exception as e:
        print(f"\n❌  Error: {e}")


def demo_analysis():
    """Demo analysis tools."""
    print("\n" + "=" * 70)
    print(" 📊  ANALYSIS DEMO")
    print("=" * 70 + "\n")
    
    print("This demo will analyze audio with:")
    print("  • Spectrum analyzer (frequency over time)")
    print("  • Beat detection (tempo and beats)")
    print("  • Loudness metering (RMS, peak, LUFS)")
    print("  • Frequency distribution (energy per band)")
    print("  • Spectral features (centroid, bandwidth, etc.)")
    print()
    
    try:
        results = showcase_analysis(
            config.AUDIO_PATH,
            output_dir=config.OUTPUT_DIR,
            save=True
        )
        print("\n✅  Analysis demo completed!")
        print(f"📁  Check {config.OUTPUT_DIR}/ for output files")
    except Exception as e:
        print(f"\n❌  Error: {e}")


def run_all_demos():
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" 🚀  RUNNING ALL DEMOS")
    print("=" * 70 + "\n")
    
    print("This will take a few minutes...")
    print()
    
    demo_studio_effects()
    print("\n" + "-" * 70 + "\n")
    
    demo_synthesis()
    print("\n" + "-" * 70 + "\n")
    
    demo_mixing()
    print("\n" + "-" * 70 + "\n")
    
    demo_analysis()
    
    print("\n" + "=" * 70)
    print(" ✅  ALL DEMOS COMPLETED!")
    print("=" * 70 + "\n")
    print(f"📁  All output files saved to: {config.OUTPUT_DIR}/")


if __name__ == "__main__":
    main()

