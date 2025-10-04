"""
MULTIMEDIA PROCESSING PLAYGROUND
--------------------------------
A collection of tools for manipulating audio, video, and images.
This script provides a variety of functions to experiment with multimedia content for social media.
"""

# Import necessary libraries
# ------------------------------

from scipy.io import wavfile
from scipy import signal
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import yt_dlp
import numpy as np
import library as lib

from IPython.display import Audio, display, Video, HTML






# If you want both video and audio downloads as separate functions:
def download_audio(url, output_dir="download/", start_time=None, duration=None):
    ydl_options = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",  # 128kbps: low quality, 320kbps: high quality
            }
        ],
    }
    if start_time is not None and duration is not None:
        ydl_options["external_downloader"] = "ffmpeg"
        ffmpeg_args = []
        if start_time is not None:
            ffmpeg_args.extend(["-ss", str(start_time)])
        if duration is not None:
            ffmpeg_args.extend(["-t", str(duration)])
        ydl_options["external_downloader_args"] = {"ffmpeg": ffmpeg_args}

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download([url])


def download_video(url, output_dir="download/", start_time=None, duration=None):
    ydl_options = {
        "format": "bestvideo+bestaudio/best",  # Gets best video+audio quality, falls back to best single format
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",
        "merge_output_format": "mp4",  # Automatically merges video and audio
    }
    if start_time is not None and duration is not None:
        ydl_options["external_downloader"] = "ffmpeg"
        ffmpeg_args = []
        if start_time is not None:
            ffmpeg_args.extend(["-ss", str(start_time)])
        if duration is not None:
            ffmpeg_args.extend(["-t", str(duration)])
        ydl_options["external_downloader_args"] = {"ffmpeg": ffmpeg_args}

    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download([url])





download_video(apple_struggle)
download_audio(dhivara_malayalam)

apple_struggle = "https://youtu.be/vN4U5FqrOdQ"


firestorm_hindi = "https://youtu.be/701FwxtMdpw"
firestorm_telegu = "https://youtu.be/FbXOsVByKmk"

satisfya_eng = "https://youtu.be/-_zKab2r3fw?list=PL5z9QulhxYohPKSJoW33ogvS6mM0RXknG"
satisfya_hindi = "https://youtu.be/pfVODjDBFxU"

mayabini_hindi = "https://youtu.be/jX9KIpErgQg"
mayabini_english = "https://youtu.be/gWhGA4JSGEk"
mayabini_assamese = "https://youtu.be/xreNppGG1lM"

dhivara_eng = "https://youtu.be/696LdKpzpUA"
dhivara_telegu = "https://youtu.be/6jpcUd5A5Jw"
dhivara_hindi = "https://youtu.be/_2clW8Zxq88"
dhivara_tamil = "https://youtu.be/YAMNFIj9NwU"
dhivara_malayalam = "https://youtu.be/aSMT7c7Q3Rc"

AikonBaikon_eng = "https://youtu.be/rx0vmM93QDs"
AikonBaikon_assamese = "https://youtu.be/sGfmDgiYAL0"

Shadowborn_portuguese = "https://youtu.be/yhg2SZp7U0k?list=RDyhg2SZp7U0k"

###############################
# AUDIO PROCESSING FUNCTIONS
###############################

AUDIO_PATH = "download/Dheevara (English Version) Full Video Song [4K]｜ Baahubali (Telugu) ｜ Prabhas, Tamannaah.mp3"  # Path to your audio file
OUTPUT_DIR = "output"

# MASHUP


def basic_audio_info():
    """Display basic information about the audio file and visualize waveform and spectrogram."""
    try:
        # Load audio file
        y, sr = librosa.load(AUDIO_PATH)

        # Print basic info
        duration = librosa.get_duration(y=y, sr=sr)
        print(f"Sample rate: {sr} Hz")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Number of samples: {len(y)}")

        # Create figure for visualizations
        plt.figure(figsize=(14, 10))

        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title("Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        # Plot spectrogram
        plt.subplot(2, 1, 2)
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Spectrogram")

        plt.tight_layout()
        plt.show()

        return y, sr
    except Exception as e:
        print(f"Error loading audio: {e}")
        return None, None


basic_audio_info()


def create_audio_mashup(y=None, sr=None, save=True):
    """Create a mashup of audio segments with different effects."""
    if y is None or sr is None:
        y, sr = librosa.load(AUDIO_PATH)

    # Split audio into segments
    segment_length = len(y) // 5
    segments = []

    for i in range(5):
        start = i * segment_length
        end = start + segment_length
        segment = y[start:end]
        segments.append(segment)

    # Apply different effects to each segment
    modified_segments = []

    # 1. Normal segment
    modified_segments.append(segments[0])

    # 2. Speed up segment
    modified_segments.append(librosa.effects.time_stretch(segments[1], rate=1.5))

    # 3. Pitch shifted segment
    modified_segments.append(librosa.effects.pitch_shift(segments[2], sr=sr, n_steps=4))

    # 4. Reversed segment
    modified_segments.append(segments[3][::-1])

    # 5. Echo effect segment
    echo_segment = np.zeros_like(segments[4])
    echo_segment[: len(segments[4])] = segments[4]
    decay = 0.5
    delay_samples = int(0.2 * sr)

    for i in range(1, 5):
        if delay_samples * i < len(echo_segment):
            echo = segments[4] * (decay**i)
            max_copy = min(len(echo), len(echo_segment) - delay_samples * i)
            echo_segment[delay_samples * i : delay_samples * i + max_copy] += echo[
                :max_copy
            ]

    echo_segment = echo_segment / np.max(np.abs(echo_segment))  # Normalize
    modified_segments.append(echo_segment)

    # Combine segments
    mashup = np.concatenate(modified_segments)

    # Normalize mashup
    mashup = mashup / np.max(np.abs(mashup))

    # Visualize mashup
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

    if save:
        sf.write(f"{OUTPUT_DIR}/audio_mashup.wav", mashup, sr)
        print(f"Mashup saved to {OUTPUT_DIR}/audio_mashup.wav")

    return mashup


create_audio_mashup()
lib.Audio(filename="output/audio_mashup.wav")




def create_voice_changer(y=None, sr=None, save=True):
    """Apply voice changing effects to make interesting voice transformations."""
    if y is None or sr is None:
        y, sr = librosa.load(AUDIO_PATH)

    voice_effects = {}

    # 1. Chipmunk voice (high pitch)
    def chipmunk(y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=7)

    voice_effects["Chipmunk"] = chipmunk(y, sr)

    # 2. Deep voice (low pitch)
    def deep_voice(y, sr):
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-6)

    voice_effects["Deep Voice"] = deep_voice(y, sr)

    y, sr = librosa.load(AUDIO_PATH)
    segment_length = len(y) // 5
    segment = y[
        2 * segment_length + (segment_length // 4) : 2 * segment_length
        + segment_length
        + (segment_length // 2)
    ]
    fast_segment = librosa.effects.time_stretch(segment, rate=1.25)
    nightcore_segment = librosa.effects.pitch_shift(fast_segment, sr=sr, n_steps=2)
    """ n_steps = -3 for male """
    # reverse_segment  = segment[::-1]
    echo_segment = (
        conv := np.convolve(
            nightcore_segment,
            np.bincount(
                np.arange(5) * int(0.2 * sr),
                weights=0.5 ** np.arange(5),
                minlength=int(0.2 * sr) * 4 + 1,
            ),
        )[: len(nightcore_segment)]
    ) / np.max(np.abs(conv))
    normalizeD_segment = echo_segment / np.max(np.abs(echo_segment))
    display(Audio(normalizeD_segment, rate=sr, autoplay=True))
    librosa.display.waveshow(normalizeD_segment, sr=sr)
    sf.write(f"{OUTPUT_DIR}/normalizeD_segment.wav", normalizeD_segment, sr)

    # 3. Robot voice (vocoder-like)
    def robot_voice(y, sr):
        # Simple vocoder-like effect
        # Extract pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

        # Create a simple "robotic" carrier signal
        hopped_y = librosa.util.frame(y, frame_length=2048, hop_length=512)
        robot = np.zeros_like(y)

        # Create a modulated sine wave based on pitch
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr
        )
        f0 = np.nan_to_num(f0)
        t = np.arange(len(y)) / sr

        carrier = np.sin(2 * np.pi * 200 * t)  # Base carrier frequency

        # Modulate with original audio
        robot = carrier * y

        # Normalize
        return robot / np.max(np.abs(robot))

    voice_effects["Robot"] = robot_voice(y, sr)

    # 4. Helium voice (very high pitch + faster)
    def helium_voice(y, sr):
        # Speed up and raise pitch
        stretched = librosa.effects.time_stretch(y, rate=1.25)
        return librosa.effects.pitch_shift(stretched, sr=sr, n_steps=12)

    voice_effects["Helium"] = helium_voice(y, sr)

    # 5. Underwater effect
    def underwater(y, sr):
        # Apply lowpass filter
        b, a = signal.butter(4, 0.2, "lowpass")
        filtered = signal.filtfilt(b, a, y)

        # Add reverb (simple implementation)
        reverb = np.zeros_like(filtered)
        delay_samples = int(0.05 * sr)  # 50ms delay
        reverb[delay_samples:] = filtered[:-delay_samples] * 0.5

        result = filtered + reverb
        return result / np.max(np.abs(result))

    voice_effects["Underwater"] = underwater(y, sr)

    # Visualize effects
    plt.figure(figsize=(15, 12))

    # Plot original waveform
    plt.subplot(3, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title("Original Voice")

    # Plot each effect
    for i, (name, effect) in enumerate(voice_effects.items(), 2):
        plt.subplot(3, 2, i)
        librosa.display.waveshow(effect, sr=sr)
        plt.title(name)

        if save:
            sf.write(
                f"{OUTPUT_DIR}/voice_{name.lower().replace(' ', '_')}.wav", effect, sr
            )
            print(
                f"Voice effect saved to {OUTPUT_DIR}/voice_{name.lower().replace(' ', '_')}.wav"
            )

    plt.tight_layout()
    plt.show()

    return voice_effects


create_voice_changer()
lib.Audio(filename="output/voice_chipmunk.wav")
lib.Audio(filename="output/voice_deep_voice.wav")
lib.Audio(filename="output/voice_robot.wav")
lib.Audio(filename="output/voice_helium.wav")
lib.Audio(filename="output/voice_underwater.wav")


"""
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################
"""
