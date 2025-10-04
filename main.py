"""
MULTIMEDIA PROCESSING PLAYGROUND - MAIN SCRIPT
-----------------------------------------------
A collection of tools for manipulating audio, video, and images.
This script orchestrates the various multimedia processing functions.
Hello Moto
"""

import sys
import os
import librosa
import soundfile as sf

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import config
from scripts import library as lib
from scripts.downloader import download_audio, download_video
from scripts.audio_info import basic_audio_info
from scripts.audio_effects import create_audio_mashup
from scripts.voice_effects import create_voice_changer, create_nightcore_effect
from scripts.daw_effects import apply_highpass_filter, apply_lowpass_filter
from scripts.daw_effects import create_daw_effects_showcase
from scripts.synthesis import showcase_synthesis
from scripts.mixing import showcase_mixing
from scripts.analysis import showcase_analysis

'''''''''''''''''''''''''''''
DOWNLOAD MODE
'''''''''''''''''''''''''''''
download_video(config.satisfya_eng, output_dir=config.DOWNLOAD_DIR)
download_audio(config.firestorm_telegu, output_dir=config.DOWNLOAD_DIR)
# lib.Audio(filename=f"{config.OUTPUT_DIR}/audio_mashup.wav")

'''''''''''''''''''''''''''''
LAB MODE
'''''''''''''''''''''''''''''
y, sr = basic_audio_info(config.AUDIO_PATH)
mashup = create_audio_mashup(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, y=y, sr=sr)
voice_effects = create_voice_changer(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, y=y, sr=sr)


y, sr = librosa.load(config.AUDIO_PATH, mono=True, duration=120.0)
print("   filtered = apply_highpass_filter(audio, sr, cutoff_freq=500, mix=1.0)")
highpass_results = apply_highpass_filter(y, sr, cutoff_freq=500, order=4, mix=1.0)
lowpass_results = apply_lowpass_filter(y, sr, cutoff_freq=3000, order=4, mix=1.0)
sf.write(f"{config.OUTPUT_DIR}/highpass_results.wav", highpass_results, sr)
sf.write(f"{config.OUTPUT_DIR}/lowpass_results.wav", lowpass_results, sr)


print("  • Compression (dynamic range control)")
print("  • Parametric EQ (frequency shaping)")
print("  • High Pass Filter (remove low frequencies, with intensity control)")
print("  • Low Pass Filter (remove high frequencies, with intensity control)")
print("  • Reverb (space simulation)")
print("  • Delay/Echo (time-based effect)")
print("  • Chorus (modulation effect)")
print("  • Flanger (jet plane effect)")
print("  • Distortion (harmonic saturation)")
print("  • Limiter (peak control)")
print("  • Gate (noise reduction)")
effects = create_daw_effects_showcase(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, save=True)

print("  • Oscillators (sine, square, sawtooth, triangle)")
print("  • ADSR envelopes (attack, decay, sustain, release)")
print("  • LFO modulation (tremolo effect)")
print("  • Chord synthesis")
print("  • Melody sequencing")
results = showcase_synthesis(output_dir=config.OUTPUT_DIR, save=True)

print("  • Multi-track mixing with gain and pan")
print("  • Crossfading between tracks")
print("  • Auto-ducking (sidechain-like)")
print("  • Stereo width enhancement")
results = showcase_mixing(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, save=True)

print("  • Spectrum analyzer (frequency over time)")
print("  • Beat detection (tempo and beats)")
print("  • Loudness metering (RMS, peak, LUFS)")
print("  • Frequency distribution (energy per band)")
print("  • Spectral features (centroid, bandwidth, etc.)")
results = showcase_analysis(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, save=True)