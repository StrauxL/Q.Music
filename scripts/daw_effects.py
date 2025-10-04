"""
DAW EFFECTS MODULE
------------------
Professional audio effects commonly found in DAWs like Ableton, Logic, etc.
Includes compression, EQ, reverb, delay, chorus, and more.
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt


def apply_compressor(y, sr, threshold_db=-20, ratio=4, attack_time=0.005, release_time=0.1):
    """
    Apply dynamic range compression to audio.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        threshold_db (float): Threshold in dB
        ratio (float): Compression ratio (e.g., 4:1)
        attack_time (float): Attack time in seconds
        release_time (float): Release time in seconds
        
    Returns:
        np.array: Compressed audio
    """
    # Convert to dB
    envelope = np.abs(y)
    envelope_db = 20 * np.log10(envelope + 1e-10)
    
    # Calculate gain reduction
    gain_reduction_db = np.where(
        envelope_db > threshold_db,
        (envelope_db - threshold_db) * (1 - 1/ratio),
        0
    )
    
    # Apply smoothing with attack and release
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    smoothed_gain = np.zeros_like(gain_reduction_db)
    for i in range(1, len(gain_reduction_db)):
        if gain_reduction_db[i] > smoothed_gain[i-1]:
            # Attack
            alpha = 1 - np.exp(-1 / attack_samples)
        else:
            # Release
            alpha = 1 - np.exp(-1 / release_samples)
        
        smoothed_gain[i] = alpha * gain_reduction_db[i] + (1 - alpha) * smoothed_gain[i-1]
    
    # Apply gain reduction
    gain_linear = 10 ** (-smoothed_gain / 20)
    compressed = y * gain_linear
    
    # Normalize
    compressed = compressed / np.max(np.abs(compressed))
    
    return compressed


def apply_parametric_eq(y, sr, freq=1000, gain_db=0, q=1.0, filter_type='peak'):
    """
    Apply parametric EQ (equalizer) to audio.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        freq (float): Center frequency in Hz
        gain_db (float): Gain in dB (for peak/shelf filters)
        q (float): Q factor (bandwidth)
        filter_type (str): 'peak', 'lowshelf', 'highshelf', 'lowpass', 'highpass'
        
    Returns:
        np.array: EQ'd audio
    """
    nyquist = sr / 2
    norm_freq = freq / nyquist
    
    if filter_type == 'peak':
        # Peaking filter
        b, a = signal.iirpeak(norm_freq, Q=q, fs=sr)
        if gain_db != 0:
            gain_linear = 10 ** (gain_db / 20)
            b = b * gain_linear
    
    elif filter_type == 'lowshelf':
        b, a = signal.butter(2, norm_freq, btype='low')
        gain_linear = 10 ** (gain_db / 20)
        b = b * gain_linear
    
    elif filter_type == 'highshelf':
        b, a = signal.butter(2, norm_freq, btype='high')
        gain_linear = 10 ** (gain_db / 20)
        b = b * gain_linear
    
    elif filter_type == 'lowpass':
        b, a = signal.butter(4, norm_freq, btype='low')
    
    elif filter_type == 'highpass':
        b, a = signal.butter(4, norm_freq, btype='high')
    
    else:
        raise ValueError(f"Unknown filter type: {filter_type}")
    
    # Apply filter
    filtered = signal.filtfilt(b, a, y)
    
    return filtered


def apply_reverb(y, sr, room_size=0.5, damping=0.5, wet_level=0.3):
    """
    Apply reverb effect using convolution with impulse response.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        room_size (float): Size of room (0-1)
        damping (float): High frequency damping (0-1)
        wet_level (float): Mix level of reverb (0-1)
        
    Returns:
        np.array: Audio with reverb
    """
    # Create a simple impulse response
    reverb_time = room_size * 2.0  # seconds
    num_samples = int(reverb_time * sr)
    
    # Generate exponentially decaying noise as impulse response
    decay = np.exp(-np.linspace(0, 10, num_samples))
    noise = np.random.randn(num_samples)
    ir = noise * decay
    
    # Apply damping (lowpass filter)
    if damping > 0:
        cutoff = (1 - damping) * 0.5  # Normalized frequency
        b, a = signal.butter(2, cutoff, btype='low')
        ir = signal.filtfilt(b, a, ir)
    
    # Normalize impulse response
    ir = ir / np.max(np.abs(ir))
    
    # Convolve with input
    wet = signal.fftconvolve(y, ir, mode='same')
    wet = wet / np.max(np.abs(wet))
    
    # Mix dry and wet
    output = (1 - wet_level) * y + wet_level * wet
    
    return output


def apply_delay(y, sr, delay_time=0.5, feedback=0.4, mix=0.5):
    """
    Apply delay/echo effect.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        delay_time (float): Delay time in seconds
        feedback (float): Feedback amount (0-1)
        mix (float): Dry/wet mix (0-1)
        
    Returns:
        np.array: Audio with delay
    """
    delay_samples = int(delay_time * sr)
    output = np.zeros(len(y) + delay_samples * 4)
    output[:len(y)] = y
    
    # Apply feedback delay
    for i in range(len(y), len(output)):
        if i >= delay_samples:
            output[i] = output[i - delay_samples] * feedback
    
    # Mix and normalize
    wet = output[:len(y)]
    result = (1 - mix) * y + mix * wet
    result = result / np.max(np.abs(result))
    
    return result


def apply_chorus(y, sr, rate=1.5, depth=0.002, mix=0.5):
    """
    Apply chorus effect using time-varying delay.
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        rate (float): LFO rate in Hz
        depth (float): Modulation depth in seconds
        mix (float): Dry/wet mix (0-1)
        
    Returns:
        np.array: Audio with chorus
    """
    # Create LFO (Low Frequency Oscillator)
    t = np.arange(len(y)) / sr
    lfo = np.sin(2 * np.pi * rate * t)
    
    # Convert depth to samples
    depth_samples = depth * sr
    
    # Apply time-varying delay
    output = np.zeros_like(y)
    for i in range(len(y)):
        delay = int(depth_samples * (1 + lfo[i]) / 2)
        if i >= delay:
            output[i] = y[i - delay]
    
    # Mix dry and wet
    result = (1 - mix) * y + mix * output
    
    return result


def apply_flanger(y, sr, rate=0.5, depth=0.003, feedback=0.5, mix=0.5):
    """
    Apply flanger effect (similar to chorus but with feedback).
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        rate (float): LFO rate in Hz
        depth (float): Modulation depth in seconds
        feedback (float): Feedback amount (0-1)
        mix (float): Dry/wet mix (0-1)
        
    Returns:
        np.array: Audio with flanger
    """
    # Create LFO
    t = np.arange(len(y)) / sr
    lfo = np.sin(2 * np.pi * rate * t)
    
    # Convert depth to samples
    depth_samples = depth * sr
    
    # Apply time-varying delay with feedback
    output = np.zeros_like(y)
    for i in range(len(y)):
        delay = int(depth_samples * (1 + lfo[i]) / 2)
        if i >= delay:
            output[i] = y[i] + feedback * output[i - delay]
    
    # Mix dry and wet
    result = (1 - mix) * y + mix * output
    result = result / np.max(np.abs(result))
    
    return result


def apply_distortion(y, gain=10, mix=1.0):
    """
    Apply distortion/overdrive effect.
    
    Args:
        y (np.array): Audio data
        gain (float): Drive amount
        mix (float): Dry/wet mix (0-1)
        
    Returns:
        np.array: Distorted audio
    """
    # Apply gain
    gained = y * gain
    
    # Soft clipping using tanh
    distorted = np.tanh(gained)
    
    # Mix
    result = (1 - mix) * y + mix * distorted
    
    # Normalize
    result = result / np.max(np.abs(result))
    
    return result


def apply_limiter(y, threshold_db=-3, release_time=0.01):
    """
    Apply limiter (hard limiting at threshold).
    
    Args:
        y (np.array): Audio data
        threshold_db (float): Threshold in dB
        release_time (float): Release time in seconds
        
    Returns:
        np.array: Limited audio
    """
    threshold = 10 ** (threshold_db / 20)
    
    # Simple hard limiting
    limited = np.clip(y, -threshold, threshold)
    
    return limited


def apply_gate(y, sr, threshold_db=-40, attack_time=0.001, release_time=0.1):
    """
    Apply noise gate (reduces audio below threshold).
    
    Args:
        y (np.array): Audio data
        sr (int): Sample rate
        threshold_db (float): Threshold in dB
        attack_time (float): Attack time in seconds
        release_time (float): Release time in seconds
        
    Returns:
        np.array: Gated audio
    """
    # Convert to dB
    envelope = np.abs(y)
    envelope_db = 20 * np.log10(envelope + 1e-10)
    
    # Calculate gate
    gate = np.where(envelope_db > threshold_db, 1.0, 0.0)
    
    # Apply smoothing
    attack_samples = int(attack_time * sr)
    release_samples = int(release_time * sr)
    
    smoothed_gate = np.zeros_like(gate)
    for i in range(1, len(gate)):
        if gate[i] > smoothed_gate[i-1]:
            alpha = 1 - np.exp(-1 / attack_samples)
        else:
            alpha = 1 - np.exp(-1 / release_samples)
        
        smoothed_gate[i] = alpha * gate[i] + (1 - alpha) * smoothed_gate[i-1]
    
    # Apply gate
    gated = y * smoothed_gate
    
    return gated


def apply_stereo_widener(y, width=1.5):
    """
    Apply stereo widening effect (requires stereo input).
    
    Args:
        y (np.array): Stereo audio data (shape: (2, n_samples))
        width (float): Width amount (1.0 = normal, >1.0 = wider)
        
    Returns:
        np.array: Widened stereo audio
    """
    if y.ndim == 1:
        # Mono input, convert to stereo
        y = np.array([y, y])
    
    # Extract mid and side
    mid = (y[0] + y[1]) / 2
    side = (y[0] - y[1]) / 2
    
    # Widen by increasing side signal
    side = side * width
    
    # Reconstruct stereo
    left = mid + side
    right = mid - side
    
    # Normalize
    stereo = np.array([left, right])
    stereo = stereo / np.max(np.abs(stereo))
    
    return stereo


def apply_panning(y, pan=0.0):
    """
    Apply panning to mono or stereo audio.
    
    Args:
        y (np.array): Audio data
        pan (float): Pan position (-1.0 = left, 0.0 = center, 1.0 = right)
        
    Returns:
        np.array: Panned stereo audio (shape: (2, n_samples))
    """
    # Ensure input is 1D
    if y.ndim > 1:
        y = y[0]  # Take first channel if stereo
    
    # Calculate pan coefficients (constant power panning)
    pan_rad = (pan + 1) * np.pi / 4  # Map -1..1 to 0..pi/2
    left_gain = np.cos(pan_rad)
    right_gain = np.sin(pan_rad)
    
    # Create stereo output
    left = y * left_gain
    right = y * right_gain
    
    return np.array([left, right])


def create_daw_effects_showcase(audio_path, output_dir="output", save=True):
    """
    Showcase all DAW effects on a given audio file.
    
    Args:
        audio_path (str): Path to audio file
        output_dir (str): Output directory
        save (bool): Whether to save files
        
    Returns:
        dict: Dictionary of all effects
    """
    print("🎛️  Loading audio...")
    y, sr = librosa.load(audio_path, mono=True)
    
    effects = {}
    
    print("🎚️  Applying compression...")
    effects['compressed'] = apply_compressor(y, sr, threshold_db=-20, ratio=4)
    
    print("🎛️  Applying EQ (boost 2kHz)...")
    effects['eq_boost'] = apply_parametric_eq(y, sr, freq=2000, gain_db=6, q=1.0)
    
    print("🎵  Applying reverb...")
    effects['reverb'] = apply_reverb(y, sr, room_size=0.7, wet_level=0.4)
    
    print("🔁  Applying delay...")
    effects['delay'] = apply_delay(y, sr, delay_time=0.375, feedback=0.4, mix=0.5)
    
    print("🎶  Applying chorus...")
    effects['chorus'] = apply_chorus(y, sr, rate=1.5, depth=0.002, mix=0.5)
    
    print("🌊  Applying flanger...")
    effects['flanger'] = apply_flanger(y, sr, rate=0.5, depth=0.003, feedback=0.5)
    
    print("🔥  Applying distortion...")
    effects['distortion'] = apply_distortion(y, gain=8, mix=0.7)
    
    print("🚧  Applying limiter...")
    effects['limited'] = apply_limiter(y, threshold_db=-3)
    
    print("🚪  Applying gate...")
    effects['gated'] = apply_gate(y, sr, threshold_db=-40)
    
    # Save files
    if save:
        print("\n💾  Saving effects...")
        for name, effect in effects.items():
            output_path = f"{output_dir}/daw_{name}.wav"
            sf.write(output_path, effect, sr)
            print(f"   ✓ {name} -> {output_path}")
    
    # Visualize
    print("\n📊  Creating visualization...")
    plt.figure(figsize=(16, 12))
    
    # Plot original
    plt.subplot(5, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original")
    plt.ylabel("Amplitude")
    
    # Plot effects
    for i, (name, effect) in enumerate(effects.items(), 2):
        plt.subplot(5, 2, i)
        librosa.display.waveshow(effect, sr=sr)
        plt.title(name.replace('_', ' ').title())
        plt.ylabel("Amplitude")
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/daw_effects_comparison.png", dpi=150)
    print(f"   ✓ Visualization saved to {output_dir}/daw_effects_comparison.png")
    plt.show()
    
    return effects


if __name__ == "__main__":
    # Test the effects
    print("DAW Effects Module - Test Mode")
    print("=" * 50)

