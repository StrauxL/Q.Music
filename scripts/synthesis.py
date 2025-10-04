"""
SYNTHESIS MODULE
----------------
Audio synthesis tools including oscillators, ADSR envelopes, and basic synthesizers.

This module provides beginner-friendly functions to create and synthesize audio
from scratch. It's perfect for learning how synthesizers work and creating
your own electronic music.

Key Concepts for Beginners:
- Oscillator: A device that generates sound waves (sine, square, sawtooth, triangle)
- ADSR Envelope: Controls how a sound starts, sustains, and ends (Attack, Decay, Sustain, Release)
- LFO: Low Frequency Oscillator used for modulation effects
- Synthesizer: An electronic instrument that creates sounds
- Frequency: How high or low a sound is (measured in Hz)
- Amplitude: How loud a sound is

Author: Q.Wave Team
"""

# Import necessary libraries for audio synthesis and visualization
import numpy as np         # For numerical operations and arrays
import matplotlib.pyplot as plt  # For creating plots and graphs
import soundfile as sf     # For reading and writing audio files
import librosa.display     # For audio visualization


class Oscillator:
    """
    Basic oscillator with multiple waveforms.
    
    An oscillator is the heart of any synthesizer - it generates the basic sound waves.
    Think of it like the strings on a guitar, but for electronic music.
    
    This class can generate different types of waveforms:
    - Sine wave: Smooth, pure tone (like a flute)
    - Square wave: Harsh, buzzy sound (like a clarinet)
    - Sawtooth wave: Bright, sharp sound (like a violin)
    - Triangle wave: Soft, mellow sound (like a muted trumpet)
    - Noise: Random sound (like wind or static)
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the oscillator.
        
        Args:
            sample_rate (int): How many samples per second (usually 44100 Hz)
        """
        self.sr = sample_rate  # Store the sample rate for use in other methods
    
    def sine(self, frequency, duration, amplitude=1.0):
        """
        Generate sine wave.
        
        A sine wave is the purest form of sound - it has no harmonics.
        It sounds smooth and flute-like.
        
        Args:
            frequency (float): Frequency in Hz (e.g., 440 for A4 note)
            duration (float): Duration in seconds
            amplitude (float): Volume (0.0 to 1.0)
            
        Returns:
            np.array: Sine wave audio data
        """
        # Create time array: from 0 to duration, with sample_rate points per second
        t = np.linspace(0, duration, int(self.sr * duration))
        # Generate sine wave: amplitude * sin(2π * frequency * time)
        return amplitude * np.sin(2 * np.pi * frequency * t)
    
    def square(self, frequency, duration, amplitude=1.0):
        """
        Generate square wave.
        
        A square wave has a harsh, buzzy sound with lots of harmonics.
        It's commonly used in electronic music and video game sounds.
        
        Args:
            frequency (float): Frequency in Hz
            duration (float): Duration in seconds
            amplitude (float): Volume (0.0 to 1.0)
            
        Returns:
            np.array: Square wave audio data
        """
        # Create time array
        t = np.linspace(0, duration, int(self.sr * duration))
        # Generate square wave: amplitude * sign(sin(2π * frequency * time))
        # sign() makes the wave jump between +1 and -1
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    
    def sawtooth(self, frequency, duration, amplitude=1.0):
        """
        Generate sawtooth wave.
        
        A sawtooth wave has a bright, sharp sound with many harmonics.
        It's great for creating lead sounds and bass lines.
        
        Args:
            frequency (float): Frequency in Hz
            duration (float): Duration in seconds
            amplitude (float): Volume (0.0 to 1.0)
            
        Returns:
            np.array: Sawtooth wave audio data
        """
        # Create time array
        t = np.linspace(0, duration, int(self.sr * duration))
        # Generate sawtooth wave: amplitude * 2 * (time * frequency - floor(time * frequency + 0.5))
        # This creates a wave that ramps up and then drops sharply
        return amplitude * 2 * (t * frequency - np.floor(t * frequency + 0.5))
    
    def triangle(self, frequency, duration, amplitude=1.0):
        """
        Generate triangle wave.
        
        A triangle wave has a soft, mellow sound with fewer harmonics than square or sawtooth.
        It's good for creating gentle, musical sounds.
        
        Args:
            frequency (float): Frequency in Hz
            duration (float): Duration in seconds
            amplitude (float): Volume (0.0 to 1.0)
            
        Returns:
            np.array: Triangle wave audio data
        """
        # Create time array
        t = np.linspace(0, duration, int(self.sr * duration))
        # Generate triangle wave by modifying a sawtooth wave
        saw = 2 * (t * frequency - np.floor(t * frequency + 0.5))  # Create sawtooth
        return amplitude * 2 * np.abs(saw) - 1  # Convert to triangle by taking absolute value
    
    def noise(self, duration, amplitude=1.0, noise_type='white'):
        """
        Generate noise.
        
        Noise is random sound that can be used for percussion, wind effects, or as a base for other sounds.
        
        Args:
            duration (float): Duration in seconds
            amplitude (float): Volume (0.0 to 1.0)
            noise_type (str): Type of noise - 'white' or 'pink'
            
        Returns:
            np.array: Noise audio data
        """
        n_samples = int(self.sr * duration)  # Calculate number of samples needed
        
        if noise_type == 'white':
            # White noise: equal energy at all frequencies (like TV static)
            return amplitude * np.random.randn(n_samples)
        elif noise_type == 'pink':
            # Pink noise: more energy at lower frequencies (like ocean waves)
            # Simple pink noise approximation using a filter
            white = np.random.randn(n_samples)
            # Filter coefficients for pink noise
            b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
            a = [1, -2.494956002, 2.017265875, -0.522189400]
            from scipy import signal
            pink = signal.lfilter(b, a, white)  # Apply the filter
            return amplitude * pink / np.max(np.abs(pink))  # Normalize
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")


class ADSR:
    """
    ADSR Envelope (Attack, Decay, Sustain, Release).
    
    An ADSR envelope controls how a sound starts, sustains, and ends.
    It's like the volume control for a note, but automated.
    
    ADSR stands for:
    - Attack: How quickly the sound reaches full volume when you press a key
    - Decay: How quickly the sound drops from full volume to the sustain level
    - Sustain: The volume level the sound stays at while you hold the key
    - Release: How quickly the sound fades to silence when you release the key
    
    Think of it like pressing a piano key - the attack is how quickly the hammer hits,
    decay is how the sound settles, sustain is the steady volume, and release is how
    it fades when you lift your finger.
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the ADSR envelope generator.
        
        Args:
            sample_rate (int): How many samples per second (usually 44100 Hz)
        """
        self.sr = sample_rate  # Store the sample rate for use in other methods
    
    def generate(self, attack=0.1, decay=0.1, sustain_level=0.7, release=0.3, duration=1.0):
        """
        Generate ADSR envelope.
        
        This creates a volume envelope that shapes how a sound starts, sustains, and ends.
        The envelope is applied to the oscillator output to create musical sounds.
        
        Args:
            attack (float): Attack time in seconds (how quickly sound reaches full volume)
            decay (float): Decay time in seconds (how quickly sound drops to sustain level)
            sustain_level (float): Sustain level (0-1, how loud the sound stays)
            release (float): Release time in seconds (how quickly sound fades to silence)
            duration (float): Total duration in seconds
            
        Returns:
            np.array: Envelope values (0-1) that can be multiplied with audio
            
        Example:
            envelope = adsr.generate(attack=0.1, decay=0.2, sustain_level=0.5, release=0.5)
            # Creates a sound that quickly attacks, slowly decays, sustains at 50% volume,
            # and slowly releases
        """
        # Calculate how many samples each phase needs
        total_samples = int(self.sr * duration)      # Total samples for entire duration
        attack_samples = int(self.sr * attack)       # Samples for attack phase
        decay_samples = int(self.sr * decay)         # Samples for decay phase
        release_samples = int(self.sr * release)     # Samples for release phase
        sustain_samples = total_samples - attack_samples - decay_samples - release_samples  # Remaining samples for sustain
        
        # Phase 1: Attack - sound goes from 0 to full volume
        # This creates a smooth ramp from 0 to 1
        attack_env = np.linspace(0, 1, attack_samples)
        
        # Phase 2: Decay - sound drops from full volume to sustain level
        # This creates a smooth ramp from 1 to sustain_level
        decay_env = np.linspace(1, sustain_level, decay_samples)
        
        # Phase 3: Sustain - sound stays at constant volume
        # This creates a flat line at the sustain level
        sustain_env = np.ones(sustain_samples) * sustain_level
        
        # Phase 4: Release - sound fades from sustain level to silence
        # This creates a smooth ramp from sustain_level to 0
        release_env = np.linspace(sustain_level, 0, release_samples)
        
        # Combine all phases into one envelope
        envelope = np.concatenate([attack_env, decay_env, sustain_env, release_env])
        
        return envelope


class LFO:
    """
    Low Frequency Oscillator for modulation.
    
    An LFO (Low Frequency Oscillator) is used to modulate (change) other parameters
    of a synthesizer. It oscillates at frequencies too low to hear (usually 0.1-20 Hz),
    but it can control things like volume (tremolo), pitch (vibrato), or filter cutoff.
    
    Think of it like a slow, invisible wave that gently changes how your sound sounds.
    For example, if you use an LFO to modulate volume, you get a tremolo effect
    (like the sound is pulsing). If you use it to modulate pitch, you get vibrato
    (like the sound is wobbling slightly).
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the LFO generator.
        
        Args:
            sample_rate (int): How many samples per second (usually 44100 Hz)
        """
        self.sr = sample_rate  # Store the sample rate for use in other methods
    
    def generate(self, frequency, duration, waveform='sine', depth=1.0):
        """
        Generate LFO signal.
        
        This creates a slow oscillation that can be used to modulate other parameters.
        The LFO runs at frequencies too low to hear, but it creates interesting effects
        when used to control other parts of the synthesizer.
        
        Args:
            frequency (float): LFO frequency in Hz (usually 0.1-20 Hz)
            duration (float): Duration in seconds
            waveform (str): Waveform type ('sine', 'triangle', 'square', 'sawtooth')
            depth (float): Modulation depth (0-1, how much the LFO affects the target)
            
        Returns:
            np.array: LFO signal (values between 0 and 1)
            
        Example:
            lfo = lfo.generate(frequency=5, duration=2, waveform='sine', depth=0.5)
            # Creates a 5 Hz sine wave LFO that modulates at 50% depth
        """
        # Create an oscillator to generate the LFO waveform
        osc = Oscillator(self.sr)
        
        # Generate the LFO waveform based on the specified type
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
        
        # Normalize to 0-1 range for easier use in modulation
        # This converts the LFO from -depth to +depth range to 0 to 1 range
        lfo = (lfo + depth) / (2 * depth)
        
        return lfo


class SimpleSynth:
    """
    Simple subtractive synthesizer.
    
    A subtractive synthesizer is the most common type of synthesizer.
    It works by starting with a rich sound (like a sawtooth wave) and then
    "subtracting" frequencies using filters to create the desired sound.
    
    This synthesizer combines:
    - Oscillator: Generates the basic sound wave
    - ADSR Envelope: Controls how the sound starts, sustains, and ends
    - LFO: Adds modulation effects like tremolo or vibrato
    - Filter: Removes unwanted frequencies
    
    Think of it like a musical instrument that you can program to make
    any sound you want - from piano-like sounds to electronic bleeps and bloops.
    """
    
    def __init__(self, sample_rate=44100):
        """
        Initialize the synthesizer.
        
        Args:
            sample_rate (int): How many samples per second (usually 44100 Hz)
        """
        self.sr = sample_rate        # Store the sample rate
        self.osc = Oscillator(sample_rate)  # Create an oscillator for generating sound waves
        self.adsr = ADSR(sample_rate)       # Create an ADSR envelope for shaping the sound
        self.lfo = LFO(sample_rate)         # Create an LFO for modulation effects
    
    def play_note(self, note='A4', duration=1.0, waveform='sine',
                  attack=0.1, decay=0.1, sustain=0.7, release=0.3,
                  filter_cutoff=None, lfo_rate=0, lfo_depth=0):
        """
        Play a single note with the synthesizer.
        
        This is the main method for creating sounds with the synthesizer.
        It combines all the components (oscillator, envelope, LFO, filter) to create
        a complete musical note.
        
        Args:
            note (str): Note name (e.g., 'A4', 'C#5', 'F3')
            duration (float): Duration in seconds
            waveform (str): Oscillator waveform ('sine', 'square', 'sawtooth', 'triangle')
            attack (float): Attack time in seconds (how quickly sound starts)
            decay (float): Decay time in seconds (how quickly sound drops to sustain level)
            sustain (float): Sustain level (0-1, how loud the sound stays)
            release (float): Release time in seconds (how quickly sound fades to silence)
            filter_cutoff (float): Lowpass filter cutoff frequency in Hz (None = no filter)
            lfo_rate (float): LFO rate in Hz (0 = no LFO modulation)
            lfo_depth (float): LFO modulation depth (0-1, how much LFO affects the sound)
            
        Returns:
            np.array: Synthesized audio data
            
        Example:
            # Play a C4 note with a sawtooth wave, quick attack, and tremolo effect
            audio = synth.play_note('C4', duration=2.0, waveform='sawtooth',
                                  attack=0.05, decay=0.1, sustain=0.7, release=0.5,
                                  lfo_rate=5, lfo_depth=0.3)
        """
        # Step 1: Convert note name to frequency
        # This converts musical note names (like 'A4') to frequencies (like 440 Hz)
        frequency = librosa.note_to_hz(note)
        
        # Step 2: Generate the basic sound wave using the oscillator
        # Choose the waveform type and generate the audio
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
        
        # Step 3: Apply LFO modulation if specified
        # This adds effects like tremolo (volume modulation) or vibrato (pitch modulation)
        if lfo_rate > 0:
            lfo_signal = self.lfo.generate(lfo_rate, duration, depth=lfo_depth)
            # Apply LFO to the audio (creates tremolo effect)
            audio = audio * (1 + lfo_signal)
        
        # Step 4: Apply ADSR envelope to shape the sound
        # This controls how the sound starts, sustains, and ends
        envelope = self.adsr.generate(attack, decay, sustain, release, duration)
        audio = audio * envelope  # Multiply audio by envelope to shape the volume
        
        # Step 5: Apply filter if specified
        # This removes unwanted frequencies to create different timbres
        if filter_cutoff is not None:
            from scipy import signal
            nyquist = self.sr / 2  # Nyquist frequency (half the sample rate)
            norm_cutoff = filter_cutoff / nyquist  # Normalize cutoff frequency
            b, a = signal.butter(4, norm_cutoff, btype='low')  # Create 4th order lowpass filter
            audio = signal.filtfilt(b, a, audio)  # Apply filter in both directions
        
        # Step 6: Normalize the audio to prevent clipping
        # This ensures the audio doesn't get too loud and cause distortion
        audio = audio / np.max(np.abs(audio))
        
        return audio
    
    def play_chord(self, notes, duration=2.0, waveform='sine',
                   attack=0.1, decay=0.1, sustain=0.7, release=0.3):
        """
        Play multiple notes simultaneously (chord).
        
        This method plays multiple notes at the same time to create a chord.
        It's like playing multiple keys on a piano at once.
        
        Args:
            notes (list): List of note names (e.g., ['C4', 'E4', 'G4'] for C major)
            duration (float): Duration in seconds
            waveform (str): Oscillator waveform ('sine', 'square', 'sawtooth', 'triangle')
            attack (float): Attack time in seconds
            decay (float): Decay time in seconds
            sustain (float): Sustain level (0-1)
            release (float): Release time in seconds
            
        Returns:
            np.array: Synthesized chord audio data
            
        Example:
            # Play a C major chord
            chord = synth.play_chord(['C4', 'E4', 'G4'], duration=3.0, waveform='sawtooth')
        """
        chord_audio = None  # Will store the combined chord audio
        
        # Step 1: Generate each note in the chord
        for note in notes:
            # Create each individual note
            note_audio = self.play_note(note, duration, waveform,
                                       attack, decay, sustain, release)
            
            # Step 2: Add this note to the chord
            if chord_audio is None:
                # First note - just store it
                chord_audio = note_audio
            else:
                # Subsequent notes - add them together
                chord_audio += note_audio
        
        # Step 3: Normalize the chord to prevent clipping
        # First, divide by the number of notes to keep the volume reasonable
        chord_audio = chord_audio / len(notes)
        # Then, normalize to prevent any clipping
        chord_audio = chord_audio / np.max(np.abs(chord_audio))
        
        return chord_audio
    
    def play_sequence(self, notes, note_duration=0.5, waveform='sine',
                     attack=0.05, decay=0.1, sustain=0.7, release=0.2):
        """
        Play a sequence of notes (melody).
        
        This method plays notes one after another to create a melody.
        It's like playing a sequence of keys on a piano in order.
        
        Args:
            notes (list): List of note names (e.g., ['C4', 'D4', 'E4', 'F4'])
            note_duration (float): Duration of each note in seconds
            waveform (str): Oscillator waveform ('sine', 'square', 'sawtooth', 'triangle')
            attack (float): Attack time in seconds
            decay (float): Decay time in seconds
            sustain (float): Sustain level (0-1)
            release (float): Release time in seconds
            
        Returns:
            np.array: Synthesized sequence audio data
            
        Example:
            # Play a simple melody
            melody = synth.play_sequence(['C4', 'E4', 'G4', 'C5'], note_duration=0.4)
        """
        sequence = []  # Will store each note in the sequence
        
        # Step 1: Generate each note in the sequence
        for note in notes:
            # Create each individual note
            note_audio = self.play_note(note, note_duration, waveform,
                                       attack, decay, sustain, release)
            # Add this note to our sequence
            sequence.append(note_audio)
        
        # Step 2: Concatenate all notes to create the complete melody
        # This joins all the notes end-to-end to create one continuous audio stream
        sequence_audio = np.concatenate(sequence)
        
        return sequence_audio


def showcase_synthesis(output_dir="output", save=True):
    """
    Showcase synthesis capabilities.
    
    This function demonstrates all the different features of the synthesizer
    by creating various sounds and saving them as audio files. It's perfect
    for learning how different parameters affect the sound.
    
    What it creates:
    1. Different waveforms (sine, square, sawtooth, triangle)
    2. Different ADSR envelopes (pluck, pad, stab)
    3. LFO modulation effects (tremolo)
    4. Chords (C major)
    5. Melody sequences
    6. Visualizations of the waveforms
    
    Args:
        output_dir (str): Directory to save output files
        save (bool): Whether to save the generated audio files
        
    Returns:
        dict: Dictionary containing all the synthesized sounds
    """
    print("🎹  Synthesis Showcase")
    print("=" * 50)
    
    # Create a synthesizer instance
    synth = SimpleSynth(sample_rate=44100)
    results = {}  # Dictionary to store all the generated sounds
    
    # 1. Generate different waveforms
    # This shows how different wave shapes sound
    print("\n📊  Generating different waveforms...")
    waveforms = ['sine', 'square', 'sawtooth', 'triangle']
    for waveform in waveforms:
        # Generate a 2-second A4 note with each waveform
        audio = synth.play_note('A4', duration=2.0, waveform=waveform)
        results[f'waveform_{waveform}'] = audio
        if save:
            sf.write(f"{output_dir}/synth_{waveform}.wav", audio, synth.sr)
            print(f"   ✓ {waveform} wave saved")
    
    # 2. Demonstrate different ADSR envelopes
    # This shows how different envelope shapes affect the sound
    print("\n🎚️  Demonstrating ADSR envelopes...")
    adsr_configs = [
        ('pluck', 0.01, 0.1, 0.3, 0.2),  # Quick attack, short sustain (like plucking a string)
        ('pad', 0.5, 0.3, 0.7, 0.8),     # Slow attack, long sustain (like a pad sound)
        ('stab', 0.05, 0.1, 0.5, 0.1),   # Quick attack, quick release (like a staccato note)
    ]
    
    for name, a, d, s, r in adsr_configs:
        # Generate a C4 note with each ADSR configuration
        audio = synth.play_note('C4', duration=2.0, waveform='sawtooth',
                               attack=a, decay=d, sustain=s, release=r)
        results[f'adsr_{name}'] = audio
        if save:
            sf.write(f"{output_dir}/synth_adsr_{name}.wav", audio, synth.sr)
            print(f"   ✓ ADSR {name} saved")
    
    # 3. Apply LFO modulation
    # This creates a tremolo effect (volume modulation)
    print("\n🌊  Applying LFO modulation...")
    audio = synth.play_note('G4', duration=3.0, waveform='sawtooth',
                           lfo_rate=5, lfo_depth=0.3)  # 5 Hz tremolo at 30% depth
    results['lfo_tremolo'] = audio
    if save:
        sf.write(f"{output_dir}/synth_lfo_tremolo.wav", audio, synth.sr)
        print(f"   ✓ LFO tremolo saved")
    
    # 4. Generate a chord
    # This shows how to play multiple notes simultaneously
    print("\n🎼  Synthesizing chord (Cmaj)...")
    chord = synth.play_chord(['C4', 'E4', 'G4'], duration=3.0, waveform='sawtooth')
    results['chord_cmajor'] = chord
    if save:
        sf.write(f"{output_dir}/synth_chord_cmajor.wav", chord, synth.sr)
        print(f"   ✓ C major chord saved")
    
    # 5. Generate a melody sequence
    # This shows how to play notes in sequence
    print("\n🎵  Playing melody sequence...")
    melody_notes = ['C4', 'E4', 'G4', 'C5', 'G4', 'E4', 'C4']  # Simple melody
    melody = synth.play_sequence(melody_notes, note_duration=0.4, waveform='square')
    results['melody'] = melody
    if save:
        sf.write(f"{output_dir}/synth_melody.wav", melody, synth.sr)
        print(f"   ✓ Melody sequence saved")
    
    # 6. Create visualizations
    # This shows the actual waveform shapes
    print("\n📊  Creating waveform visualization...")
    plt.figure(figsize=(14, 10))
    
    for i, waveform in enumerate(waveforms, 1):
        plt.subplot(2, 2, i)  # 2x2 grid, position i
        audio = results[f'waveform_{waveform}']
        plt.plot(audio[:2000])  # Plot first 2000 samples
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

