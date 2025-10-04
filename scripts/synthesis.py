"""
SYNTHESIS MODULE
----------------
Audio synthesis tools including oscillators, ADSR envelopes, and basic synthesizers.
"""

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa.display


class Oscillator:
    """
    Basic oscillator with multiple waveforms.
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def sine(self, frequency, duration, amplitude=1.0):
        """Generate sine wave."""
        t = np.linspace(0, duration, int(self.sr * duration))
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def square(self, frequency, duration, amplitude=1.0):
        """Generate square wave."""
        t = np.linspace(0, duration, int(self.sr * duration))
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    
    def sawtooth(self, frequency, duration, amplitude=1.0):
        """Generate sawtooth wave."""
        t = np.linspace(0, duration, int(self.sr * duration))
        return amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
    
    def triangle(self, frequency, duration, amplitude=1.0):
        """Generate triangle wave."""
        t = np.linspace(0, duration, int(self.sr * duration))
        saw = 2 * (t * frequency - np.floor(t * frequency + 0.5))
        return amplitude * 2 * np.abs(saw) - 1
    
    def noise(self, duration, amplitude=1.0, noise_type='white'):
        """Generate noise."""
        n_samples = int(self.sr * duration)
        
        if noise_type == 'white':
            return amplitude * np.random.randn(n_samples)
        elif noise_type == 'pink':
            # Simple pink noise approximation
            white = np.random.randn(n_samples)
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy import signal
            pink = signal.lfilter(b, a, white)
            return amplitude * pink / np.max(np.abs(pink))
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")


class ADSR:
    """
    ADSR Envelope (Attack, Decay, Sustain, Release).
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def generate(self, attack=0.1, decay=0.1, sustain_level=0.7, release=0.3, duration=1.0):
        """
        Generate ADSR envelope.
        
        Args:
            attack (float): Attack time in seconds
            decay (float): Decay time in seconds
            sustain_level (float): Sustain level (0-1)
            release (float): Release time in seconds
            duration (float): Total duration in seconds
            
        Returns:
            np.array: Envelope
        """
        total_samples = int(self.sr * duration)
        attack_samples = int(self.sr * attack)
        decay_samples = int(self.sr * decay)
        release_samples = int(self.sr * release)
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples
        
        # Attack: 0 to 1
        attack_env = np.linspace(0, 1, attack_samples)
        
        # Decay: 1 to sustain_level
        decay_env = np.linspace(1, sustain_level, decay_samples)
        
        # Sustain: constant at sustain_level
        sustain_env = np.ones(sustain_samples) * sustain_level
        
        # Release: sustain_level to 0
        release_env = np.linspace(sustain_level, 0, release_samples)
        
        # Concatenate all phases
        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
        
        return envelope


class LFO:
    """
    Low Frequency Oscillator for modulation.
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
    
    def generate(self, frequency, duration, waveform='sine', depth=1.0):
        """
        Generate LFO signal.
        
        Args:
            frequency (float): LFO frequency in Hz
            duration (float): Duration in seconds
            waveform (str): Waveform type ('sine', 'triangle', 'square', 'sawtooth')
            depth (float): Modulation depth (0-1)
            
        Returns:
            np.array: LFO signal
        """
        osc = Oscillator(self.sr)
        
        if waveform == 'sine':
            lfo = osc.sine(frequency, duration, amplitude=depth)
        elif waveform == 'triangle':
            lfo = osc.triangle(frequency, duration, amplitude=depth)
        elif waveform == 'square':
            lfo = osc.square(frequency, duration, amplitude=depth)
        elif waveform == 'sawtooth':
            lfo = osc.sawtooth(frequency, duration, amplitude=depth)
        else:
            raise ValueError(f"Unknown waveform: {waveform}")
        
        # Normalize to 0-1 range
        lfo = (lfo + depth) / (2 * depth)
        
        return lfo


class SimpleSynth:
    """
    Simple subtractive synthesizer.
    """
    
    def __init__(self, sample_rate=44100):
        self.sr = sample_rate
        self.osc = Oscillator(sample_rate)
        self.adsr = ADSR(sample_rate)
        self.lfo = LFO(sample_rate)
    
    def play_note(self, note='A4', duration=1.0, waveform='sine',
                  attack=0.1, decay=0.1, sustain=0.7, release=0.3,
                  filter_cutoff=None, lfo_rate=0, lfo_depth=0):
        """
        Play a single note with the synthesizer.
        
        Args:
            note (str): Note name (e.g., 'A4', 'C#5')
            duration (float): Duration in seconds
            waveform (str): Oscillator waveform
            attack, decay, sustain, release: ADSR parameters
            filter_cutoff (float): Lowpass filter cutoff frequency (Hz)
            lfo_rate (float): LFO rate in Hz (0 = no LFO)
            lfo_depth (float): LFO modulation depth
            
        Returns:
            np.array: Synthesized audio
        """
        # Convert note to frequency
        frequency = librosa.note_to_hz(note)
        
        # Generate oscillator
        if waveform == 'sine':
            audio = self.osc.sine(frequency, duration)
        elif waveform == 'square':
            audio = self.osc.square(frequency, duration)
        elif waveform == 'sawtooth':
            audio = self.osc.sawtooth(frequency, duration)
        elif waveform == 'triangle':
            audio = self.osc.triangle(frequency, duration)
        else:
            raise ValueError(f"Unknown waveform: {waveform}")
        
        # Apply LFO if specified
        if lfo_rate > 0:
            lfo_signal = self.lfo.generate(lfo_rate, duration, depth=lfo_depth)
            audio = audio * (1 + lfo_signal)
        
        # Apply ADSR envelope
        envelope = self.adsr.generate(attack, decay, sustain, release, duration)
        audio = audio * envelope
        
        # Apply filter if specified
        if filter_cutoff is not None:
            from scipy import signal
            nyquist = self.sr / 2
            norm_cutoff = filter_cutoff / nyquist
            b, a = signal.butter(4, norm_cutoff, btype='low')
            audio = signal.filtfilt(b, a, audio)
        
        # Normalize
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def play_chord(self, notes, duration=2.0, waveform='sine',
                   attack=0.1, decay=0.1, sustain=0.7, release=0.3):
        """
        Play multiple notes simultaneously (chord).
        
        Args:
            notes (list): List of note names
            duration (float): Duration in seconds
            waveform (str): Oscillator waveform
            attack, decay, sustain, release: ADSR parameters
            
        Returns:
            np.array: Synthesized chord
        """
        chord_audio = None
        
        for note in notes:
            note_audio = self.play_note(note, duration, waveform,
                                       attack, decay, sustain, release)
            
            if chord_audio is None:
                chord_audio = note_audio
            else:
                chord_audio += note_audio
        
        # Normalize
        chord_audio = chord_audio / len(notes)
        chord_audio = chord_audio / np.max(np.abs(chord_audio))
        
        return chord_audio
    
    def play_sequence(self, notes, note_duration=0.5, waveform='sine',
                     attack=0.05, decay=0.1, sustain=0.7, release=0.2):
        """
        Play a sequence of notes (melody).
        
        Args:
            notes (list): List of note names
            note_duration (float): Duration of each note in seconds
            waveform (str): Oscillator waveform
            attack, decay, sustain, release: ADSR parameters
            
        Returns:
            np.array: Synthesized sequence
        """
        sequence = []
        
        for note in notes:
            note_audio = self.play_note(note, note_duration, waveform,
                                       attack, decay, sustain, release)
            sequence.append(note_audio)
        
        # Concatenate all notes
        sequence_audio = np.concatenate(sequence)
        
        return sequence_audio


def showcase_synthesis(output_dir="output", save=True):
    """
    Showcase synthesis capabilities.
    
    Args:
        output_dir (str): Output directory
        save (bool): Whether to save files
        
    Returns:
        dict: Dictionary of synthesized sounds
    """
    print("🎹  Synthesis Showcase")
    print("=" * 50)
    
    synth = SimpleSynth(sample_rate=44100)
    results = {}
    
    # 1. Different waveforms
    print("\n📊  Generating different waveforms...")
    waveforms = ['sine', 'square', 'sawtooth', 'triangle']
    for waveform in waveforms:
        audio = synth.play_note('A4', duration=2.0, waveform=waveform)
        results[f'waveform_{waveform}'] = audio
        if save:
            sf.write(f"{output_dir}/synth_{waveform}.wav", audio, synth.sr)
            print(f"   ✓ {waveform} wave saved")
    
    # 2. ADSR envelope demonstration
    print("\n🎚️  Demonstrating ADSR envelopes...")
    adsr_configs = [
        ('pluck', 0.01, 0.1, 0.3, 0.2),
        ('pad', 0.5, 0.3, 0.7, 0.8),
        ('stab', 0.05, 0.1, 0.5, 0.1),
    ]
    
    for name, a, d, s, r in adsr_configs:
        audio = synth.play_note('C4', duration=2.0, waveform='sawtooth',
                               attack=a, decay=d, sustain=s, release=r)
        results[f'adsr_{name}'] = audio
        if save:
            sf.write(f"{output_dir}/synth_adsr_{name}.wav", audio, synth.sr)
            print(f"   ✓ ADSR {name} saved")
    
    # 3. LFO modulation
    print("\n🌊  Applying LFO modulation...")
    audio = synth.play_note('G4', duration=3.0, waveform='sawtooth',
                           lfo_rate=5, lfo_depth=0.3)
    results['lfo_tremolo'] = audio
    if save:
        sf.write(f"{output_dir}/synth_lfo_tremolo.wav", audio, synth.sr)
        print(f"   ✓ LFO tremolo saved")
    
    # 4. Chord
    print("\n🎼  Synthesizing chord (Cmaj)...")
    chord = synth.play_chord(['C4', 'E4', 'G4'], duration=3.0, waveform='sawtooth')
    results['chord_cmajor'] = chord
    if save:
        sf.write(f"{output_dir}/synth_chord_cmajor.wav", chord, synth.sr)
        print(f"   ✓ C major chord saved")
    
    # 5. Melody sequence
    print("\n🎵  Playing melody sequence...")
    melody_notes = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4']
    melody = synth.play_sequence(melody_notes, note_duration=0.4, waveform='square')
    results['melody'] = melody
    if save:
        sf.write(f"{output_dir}/synth_melody.wav", melody, synth.sr)
        print(f"   ✓ Melody sequence saved")
    
    # Visualize waveforms
    print("\n📊  Creating waveform visualization...")
    plt.figure(figsize=(14, 10))
    
    for i, waveform in enumerate(waveforms, 1):
        plt.subplot(2, 2, i)
        audio = results[f'waveform_{waveform}']
        plt.plot(audio[:2000])
        plt.title(f'{waveform.capitalize()} Wave')
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save:
        plt.savefig(f"{output_dir}/synth_waveforms.png", dpi=150)
        print(f"   ✓ Waveform visualization saved")
    plt.show()
    
    return results


if __name__ == "__main__":
    print("Synthesis Module - Test Mode")
    print("=" * 50)

