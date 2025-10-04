# Q.Music - Professional Audio Processing Suite

A comprehensive multimedia processing toolkit featuring professional Digital Audio Workstation (DAW) capabilities, audio synthesis, mixing tools, and creative effects for social media content creation and audio experimentation.

## 🚀 Quick Start

### Installation

1. **Activate Virtual Environment:**
```bash
source venv/bin/activate
# or use the convenience script:
./activate_venv.sh
```

2. **Verify Installation:**
```bash
python --version  # Python 3.11
python -c "import librosa; print('Ready!')"
```

### Run Interactive Demo

```bash
# Explore DAW features with interactive demos
python daw_demo.py

# Or run the original features
python main.py
```

---

## 📋 Table of Contents

- [Features Overview](#-features-overview)
- [Project Structure](#-project-structure)
- [Module Documentation](#-module-documentation)
  - [Studio Effects](#1-studio-effects)
  - [Synthesis](#2-synthesis)
  - [Mixing Tools](#3-mixing-tools)
  - [Audio Analysis](#4-audio-analysis)
  - [Original Modules](#original-modules)
- [Quick Reference](#-quick-reference)
- [Common Workflows](#-common-workflows)
- [Technical Reference](#-technical-reference)
- [Troubleshooting](#-troubleshooting)

---

## ✨ Features Overview

### Professional DAW Features
✅ **Studio Effects**: Compression, EQ, Reverb, Delay, Chorus, Flanger, Distortion, Limiter, Gate  
✅ **Audio Synthesis**: Oscillators, ADSR envelopes, LFO modulation, complete synthesizer  
✅ **Mixing Tools**: Multi-track mixing, crossfading, auto-ducking, stereo enhancement  
✅ **Audio Analysis**: Spectrum analyzer, beat detection, loudness metering, spectral features  

### Original Features
✅ Download audio/video from YouTube and other platforms  
✅ Audio analysis with waveform and spectrogram visualization  
✅ Creative audio mashups with multiple effects  
✅ Voice transformation effects (chipmunk, robot, helium, underwater, deep)  
✅ Nightcore effect creation  

---

## 📁 Project Structure

```
Q.Music/
├── main.py                      # Main orchestration script
├── daw_demo.py                  # Interactive DAW features demo
├── example_usage.py             # Example code snippets
├── activate_venv.sh             # Virtual environment activation script
├── requirements_venv.txt        # Dependencies
│
├── scripts/                     # Core functionality modules
│   ├── __init__.py
│   ├── config.py               # Configuration & URL constants
│   ├── downloader.py           # Audio/video download functions
│   ├── audio_info.py           # Audio analysis & visualization
│   ├── audio_effects.py        # Audio mashup & effects
│   ├── voice_effects.py        # Voice transformation effects
│   ├── library.py              # Utility functions
│   ├── daw_effects.py          # Professional studio effects
│   ├── synthesis.py            # Audio synthesis tools
│   ├── mixing.py               # Multi-track mixing
│   └── analysis.py             # Audio analysis tools
│
├── venv/                        # Virtual environment
├── download/                    # Downloaded content storage
├── output/                      # Processed audio output
└── input/                       # Input files
```

---

## 📚 Module Documentation

### 1. Studio Effects (`scripts/daw_effects.py`)

Professional audio effects for sound design and mixing.

#### Available Effects:

| Effect | Description | Key Parameters |
|--------|-------------|----------------|
| **Compressor** | Dynamic range control | `threshold_db`, `ratio`, `attack_time`, `release_time` |
| **Parametric EQ** | Frequency shaping | `freq`, `gain_db`, `q`, `filter_type` |
| **Reverb** | Spatial effects | `room_size`, `damping`, `wet_level` |
| **Delay/Echo** | Time-based effects | `delay_time`, `feedback`, `mix` |
| **Chorus** | Modulation effect | `rate`, `depth`, `mix` |
| **Flanger** | Jet plane effect | `rate`, `depth`, `feedback` |
| **Distortion** | Harmonic saturation | `gain`, `mix` |
| **Limiter** | Peak control | `threshold_db` |
| **Gate** | Noise reduction | `threshold_db`, `attack_time`, `release_time` |
| **Stereo Widener** | Enhance stereo image | `width` |
| **Panning** | Stereo positioning | `pan` (-1 to 1) |

#### Quick Examples:

```python
from scripts.daw_effects import *
import librosa

y, sr = librosa.load("audio.mp3")

# Compression
compressed = apply_compressor(y, sr, threshold_db=-20, ratio=4)

# EQ boost at 2kHz
boosted = apply_parametric_eq(y, sr, freq=2000, gain_db=6)

# Reverb
reverb = apply_reverb(y, sr, room_size=0.7, wet_level=0.4)

# Delay/Echo
delayed = apply_delay(y, sr, delay_time=0.5, feedback=0.4)
```

---

### 2. Synthesis (`scripts/synthesis.py`)

Generate sounds from scratch with oscillators and synthesis techniques.

#### Classes:

**Oscillator** - Generate basic waveforms
- `sine()` - Pure sine wave
- `square()` - Rich in odd harmonics
- `sawtooth()` - Rich in all harmonics
- `triangle()` - Softer than square
- `noise()` - White and pink noise

**ADSR** - Envelope generator for shaping amplitude over time

**LFO** - Low Frequency Oscillator for modulation effects

**SimpleSynth** - Complete synthesizer combining all components

#### Quick Examples:

```python
from scripts.synthesis import SimpleSynth

synth = SimpleSynth()

# Single note
note = synth.play_note('A4', duration=1.0, waveform='sawtooth')

# Chord
chord = synth.play_chord(['C4', 'E4', 'G4'], duration=2.0)

# Melody
melody = synth.play_sequence(['C4', 'D4', 'E4', 'F4', 'G4'], note_duration=0.4)

# Custom ADSR (pluck sound)
pluck = synth.play_note('E4', duration=1.0, waveform='sawtooth',
                        attack=0.01, decay=0.1, sustain=0.3, release=0.2)

# With LFO (tremolo)
tremolo = synth.play_note('G4', duration=2.0, lfo_rate=5, lfo_depth=0.3)
```

---

### 3. Mixing Tools (`scripts/mixing.py`)

Professional mixing capabilities for multi-track audio.

#### Features:

- **Mixer Class**: Multi-track mixing with gain and pan control
- **Crossfade**: Smooth transitions between tracks
- **Auto-ducking**: Sidechain-like effect (perfect for podcasts)
- **Stereo Enhancement**: Create stereo from mono

#### Quick Examples:

```python
from scripts.mixing import Mixer, crossfade, auto_duck
import librosa

# Multi-track mixing
mixer = Mixer(sample_rate=44100)
track1, sr = librosa.load("audio1.mp3")
track2, _ = librosa.load("audio2.mp3")

mixer.add_track("Drums", track1, gain_db=0, pan=-0.5)
mixer.add_track("Bass", track2, gain_db=-3, pan=0.5)
mixed = mixer.mix(normalize=True)

# Crossfade
result = crossfade(track1, track2, crossfade_duration=2.0, sr=sr)

# Auto-duck (lower music when voice plays)
ducked = auto_duck(music, voice, threshold_db=-25, ratio=4, sr=sr)
```

---

### 4. Audio Analysis (`scripts/analysis.py`)

Professional audio analysis and metering tools.

#### Analysis Tools:

- **Spectrum Analyzer**: Frequency over time visualization
- **Beat Detector**: Automatic tempo detection (BPM) and beat tracking
- **Loudness Meter**: RMS, Peak, and LUFS metering
- **Frequency Distribution**: Energy distribution across frequency bands
- **Phase Correlation**: Stereo analysis and mono compatibility
- **Spectral Features**: Timbre analysis (centroid, bandwidth, rolloff)

#### Quick Examples:

```python
from scripts.analysis import beat_detector, loudness_meter
import librosa

y, sr = librosa.load("audio.mp3")

# Beat detection
beats = beat_detector(y, sr)
print(f"Tempo: {beats['tempo']:.1f} BPM")

# Loudness
loudness = loudness_meter(y, sr, meter_type='rms')
print(f"RMS: {loudness['rms_mean']:.2f} dB")
```

---

### Original Modules

#### `config.py`
Configuration constants and URL references for various songs in multiple languages.

#### `downloader.py`
Download functions using yt-dlp:
- `download_audio()` - Download and convert to MP3
- `download_video()` - Download video in MP4 format

#### `audio_info.py`
Audio analysis and visualization:
- `basic_audio_info()` - Display waveform, spectrogram, and metadata

#### `audio_effects.py`
Audio mashup creation with 5 different effects segments.

#### `voice_effects.py`
Voice transformation effects:
- Chipmunk, Deep Voice, Robot, Helium, Underwater
- Nightcore effect

---

## 🎯 Quick Reference

### Common Imports

```python
# Studio Effects
from scripts.daw_effects import (
    apply_compressor, apply_parametric_eq, apply_reverb,
    apply_delay, apply_chorus, apply_flanger, apply_distortion,
    apply_limiter, apply_gate, apply_panning
)

# Synthesis
from scripts.synthesis import Oscillator, ADSR, LFO, SimpleSynth

# Mixing
from scripts.mixing import Mixer, crossfade, auto_duck

# Analysis
from scripts.analysis import (
    spectrum_analyzer, beat_detector, loudness_meter,
    frequency_distribution, spectral_features
)

# Original features
from scripts.downloader import download_audio, download_video
from scripts.audio_effects import create_audio_mashup
from scripts.voice_effects import create_voice_changer
```

### Parameter Ranges

| Effect | Parameter | Range | Common | Unit |
|--------|-----------|-------|--------|------|
| **Compressor** | threshold_db | -40 to 0 | -20 to -10 | dB |
| | ratio | 1.5 to 20 | 2 to 6 | :1 |
| | attack_time | 0.001 to 0.1 | 0.005 to 0.05 | sec |
| | release_time | 0.05 to 1.0 | 0.1 to 0.3 | sec |
| **EQ** | freq | 20 to 20000 | varies | Hz |
| | gain_db | -24 to +24 | -6 to +6 | dB |
| | q | 0.1 to 10 | 0.5 to 3 | - |
| **Reverb** | room_size | 0 to 1 | 0.3 to 0.7 | - |
| | damping | 0 to 1 | 0.4 to 0.6 | - |
| | wet_level | 0 to 1 | 0.2 to 0.4 | - |
| **Delay** | delay_time | 0.05 to 2.0 | 0.25 to 0.5 | sec |
| | feedback | 0 to 0.9 | 0.3 to 0.5 | - |
| **ADSR** | attack | 0.001 to 1.0 | 0.01 to 0.1 | sec |
| | decay | 0.01 to 1.0 | 0.05 to 0.2 | sec |
| | sustain | 0 to 1 | 0.5 to 0.7 | level |
| | release | 0.01 to 2.0 | 0.1 to 0.5 | sec |

### Musical Notes (A440 tuning)

```
C4 = Middle C = 261.63 Hz
A4 = Concert A = 440.00 Hz

Octaves: C0, C1, C2, C3, C4, C5, C6, C7, C8
Sharp notes: C#, D#, F#, G#, A#
Flat notes: Db, Eb, Gb, Ab, Bb

Common Chords:
- C major: ['C4', 'E4', 'G4']
- C minor: ['C4', 'Eb4', 'G4']
- C7: ['C4', 'E4', 'G4', 'Bb4']
```

### Frequency Ranges

```
Sub-bass:     20 Hz - 60 Hz
Bass:         60 Hz - 250 Hz
Low-mid:      250 Hz - 500 Hz
Midrange:     500 Hz - 2 kHz
Upper-mid:    2 kHz - 4 kHz
Presence:     4 kHz - 6 kHz
Brilliance:   6 kHz - 20 kHz
```

---

## 🎨 Common Workflows

### 1. Mastering Chain

```python
from scripts.daw_effects import *
import librosa
import soundfile as sf

y, sr = librosa.load("track.mp3")

# 1. EQ - Remove rumble
y = apply_parametric_eq(y, sr, freq=40, filter_type='highpass')

# 2. Compression
y = apply_compressor(y, sr, threshold_db=-15, ratio=3)

# 3. EQ - Add presence
y = apply_parametric_eq(y, sr, freq=3000, gain_db=3)

# 4. Limiter
y = apply_limiter(y, threshold_db=-1)

sf.write("output/mastered.wav", y, sr)
```

### 2. Vocal Processing

```python
from scripts.daw_effects import *

# 1. Gate - Remove background noise
voice = apply_gate(voice, sr, threshold_db=-45)

# 2. EQ - Cut low end
voice = apply_parametric_eq(voice, sr, freq=80, filter_type='highpass')

# 3. De-esser (EQ cut at 7kHz)
voice = apply_parametric_eq(voice, sr, freq=7000, gain_db=-4, q=3)

# 4. Compression
voice = apply_compressor(voice, sr, threshold_db=-18, ratio=4)

# 5. EQ - Add clarity
voice = apply_parametric_eq(voice, sr, freq=3000, gain_db=3)

# 6. Light reverb
voice = apply_reverb(voice, sr, room_size=0.3, wet_level=0.2)
```

### 3. Podcast Production

```python
from scripts.mixing import auto_duck
from scripts.daw_effects import *
import librosa
import soundfile as sf

# Load voice and music
voice, sr = librosa.load("voice.mp3")
music, _ = librosa.load("music.mp3")

# Process voice
voice = apply_gate(voice, sr, threshold_db=-45)
voice = apply_parametric_eq(voice, sr, freq=100, filter_type='highpass')
voice = apply_compressor(voice, sr, threshold_db=-18, ratio=4)

# Auto-duck music when voice plays
music_ducked = auto_duck(main_audio=music, ducking_audio=voice,
                         threshold_db=-30, ratio=6, sr=sr)

# Mix together
final = voice + music_ducked[:len(voice)]
final = final / np.max(np.abs(final))

sf.write("output/podcast.wav", final, sr)
```

### 4. Electronic Music Production

```python
from scripts.synthesis import SimpleSynth
from scripts.mixing import Mixer
import soundfile as sf

synth = SimpleSynth()

# Create bass line
bass = synth.play_sequence(['C2', 'C2', 'G2', 'C2'], note_duration=0.5)

# Create lead melody
lead = synth.play_note('C5', duration=2.0, waveform='square',
                       attack=0.05, lfo_rate=5, lfo_depth=0.2)

# Mix tracks
mixer = Mixer()
mixer.add_track("Bass", bass, gain_db=0, pan=0)
mixer.add_track("Lead", lead, gain_db=-3, pan=0.3)
final = mixer.mix()

sf.write("output/electronic.wav", final, synth.sr)
```

### 5. Sound Design

```python
from scripts.synthesis import SimpleSynth
from scripts.daw_effects import *

synth = SimpleSynth()

# Start with sawtooth oscillator
sound = synth.play_note('A3', duration=3.0, waveform='sawtooth',
                        attack=0.1, decay=0.3, sustain=0.6, release=0.8)

# Apply effects chain
sound = apply_parametric_eq(sound, synth.sr, freq=500, filter_type='lowpass')
sound = apply_distortion(sound, gain=5, mix=0.5)
sound = apply_chorus(sound, synth.sr, rate=2, depth=0.003)
sound = apply_delay(sound, synth.sr, delay_time=0.375, feedback=0.4)
sound = apply_reverb(sound, synth.sr, room_size=0.8, wet_level=0.3)

sf.write("output/sound_design.wav", sound, synth.sr)
```

---

## 📊 Technical Reference

### Processing Pipeline

```
Input Audio
    │
    ├─→ librosa.load() → numpy array (y, sr)
    │
    ├─→ DSP Processing (numpy/scipy)
    │   ├─ Effects (daw_effects.py)
    │   ├─ Synthesis (synthesis.py)
    │   ├─ Mixing (mixing.py)
    │   └─ Analysis (analysis.py)
    │
    └─→ soundfile.write() → Output WAV
```

### Audio Processing Concepts

- **Sample Rate**: 44.1 kHz (CD quality) by default
- **Bit Depth**: 32-bit float WAV for maximum quality
- **Stereo Processing**: Most effects support both mono and stereo
- **Normalization**: Automatic peak normalization prevents clipping

### Performance Tips

1. **Use numpy arrays**: All functions work with numpy arrays for efficiency
2. **Reuse loaded audio**: Load once, process multiple times
3. **Batch processing**: Process multiple files in loops
4. **Lower sample rate**: Use 22050 Hz for faster prototyping

### DSP Terms Glossary

- **FFT**: Fast Fourier Transform - converts time domain to frequency domain
- **STFT**: Short-Time Fourier Transform - FFT over time windows
- **dB**: Decibels - logarithmic scale for amplitude
- **Q Factor**: Bandwidth of a filter (higher Q = narrower)
- **ADSR**: Attack, Decay, Sustain, Release envelope
- **LFO**: Low Frequency Oscillator for modulation

---

## 🔧 Installation Details

### What's Installed

The virtual environment (`venv`) includes:

**Core Packages:**
- `yt-dlp` 2025.9.26 (latest version)
- `librosa` 0.11.0
- `scipy` 1.15.3
- `numpy` 2.2.5
- `matplotlib` 3.10.3
- `opencv-python` 4.11.0.86
- `soundfile` 0.13.1
- `ipython`, `jupyter` (full notebook environment)

**Tools:**
- `ffmpeg` 7.0.2 (static build in `venv/bin/ffmpeg_binaries/`)
- `ffprobe`

### Dependencies

Install from requirements file:
```bash
pip install -r requirements_venv.txt
```

Key dependencies:
- `librosa` - Audio processing
- `scipy` - Signal processing
- `matplotlib` - Visualization
- `yt-dlp` - Video/audio downloads
- `soundfile` - Audio file I/O

---

## 🛠️ Troubleshooting

### Common Issues

**Q: Effects sound distorted**  
A: Lower the input gain or adjust threshold parameters. Use normalization after processing.

**Q: Output is clipping**  
A: Enable normalization or use a limiter as the last effect in your chain.

**Q: Synthesis sounds dull**  
A: Try sawtooth/square waves instead of sine, or add harmonics with distortion.

**Q: Beat detection not working**  
A: Ensure audio has clear rhythmic content. Try adjusting beat detection parameters.

**Q: Slow processing**  
A: Use shorter audio clips for testing. Consider lowering sample rate for experimentation.

**Q: YouTube download fails**  
A: Make sure `yt-dlp` is updated to the latest version. Check internet connection.

**Q: Module import errors**  
A: Ensure virtual environment is activated: `source venv/bin/activate`

### Debugging Audio

```python
import numpy as np

# Check audio shape
print(f"Shape: {y.shape}")
print(f"Duration: {len(y)/sr:.2f}s")
print(f"Sample rate: {sr} Hz")

# Check for clipping
if np.max(np.abs(y)) > 1.0:
    print("⚠️ Audio is clipping!")
    y = y / np.max(np.abs(y))  # Normalize

# Check for NaN or Inf
if np.any(np.isnan(y)) or np.any(np.isinf(y)):
    print("⚠️ Invalid values detected!")
```

---

## 💡 Pro Tips

1. **Always normalize** after processing to prevent clipping
2. **Use EQ before compression** in mastering chains
3. **Less is more** - don't over-process your audio
4. **Listen at low volumes** to check mix balance
5. **Save intermediate steps** for A/B comparison
6. **Use reference tracks** to guide your processing decisions
7. **High-pass filter everything** except bass/kick to clean up mixes
8. **Compress in stages** rather than one heavy compressor
9. **Use reverb and delay sparingly** to avoid mud
10. **Master at -1dB** to leave headroom for conversion/streaming

---

## 🎯 Use Cases

### Music Production
- Apply professional effects to recorded instruments
- Create electronic music with synthesizers
- Mix multiple tracks with proper gain staging
- Master final tracks for distribution

### Sound Design
- Create custom sound effects from scratch
- Design synthesizer patches for games/films
- Apply creative effects for unique sounds
- Generate atmospheric soundscapes

### Podcast Production
- Auto-duck music under voice
- Remove background noise with gates
- Compress voice for consistency
- EQ for clarity and presence

### Audio Analysis
- Analyze tempo and beats for DJ mixing
- Measure loudness for broadcast standards
- Check mono compatibility for streaming
- Extract musical features for ML applications

### Educational
- Learn DSP concepts hands-on
- Understand audio effects parameters
- Experiment with synthesis techniques
- Visualize audio in various domains

---

## 🤝 Contributing

Feel free to extend this project with:
- New effects and parameters
- Better synthesis algorithms
- More analysis tools
- Optimization improvements
- Documentation and examples

---

## 📄 License

Q.Music - For educational and experimental use.

---

## 📚 Additional Resources

### Recommended Reading
1. "The Audio Programming Book" - DSP fundamentals
2. Ableton Manual - Understanding effect parameters
3. "Designing Sound" by Andy Farnell - Synthesis techniques
4. "Mixing Secrets" by Mike Senior - Mixing techniques

### Online Resources
- [Librosa Documentation](https://librosa.org/)
- [SciPy Signal Processing](https://docs.scipy.org/doc/scipy/reference/signal.html)
- [Sound on Sound](https://www.soundonsound.com/) - Audio engineering articles

---

**Happy Audio Processing! 🎵🎶🎛️🎹**

For interactive exploration:
```bash
python daw_demo.py          # Full demonstrations
python example_usage.py     # Quick code examples
```
