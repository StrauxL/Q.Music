"""
DOWNLOADER MODULE
-----------------
Functions for downloading audio and video content from online sources.

This module provides beginner-friendly functions to download audio and video
from YouTube and other online sources. It's perfect for getting content to
experiment with in your audio processing projects.

Key Concepts for Beginners:
- URL: Web address that points to a video or audio file
- Download: Getting a file from the internet to your computer
- Audio Extraction: Converting video to audio format
- Format Conversion: Changing file format (e.g., MP4 to MP3)
- Quality Settings: Choosing how good the audio/video quality should be

Author: Q.Wave Team
"""

# Import the YouTube downloader library
import yt_dlp  # This library can download from YouTube and many other sites


def download_audio(url, output_dir="download/", start_time=None, duration=None):
    """
    Download audio from a URL and convert to MP3.
    
    This function downloads audio from YouTube or other video sites and converts
    it to MP3 format. It's perfect for getting audio files to experiment with
    in your audio processing projects.
    
    What it does:
    1. Takes a YouTube URL (or other video URL)
    2. Downloads the best quality audio available
    3. Converts it to MP3 format
    4. Saves it to your specified directory
    5. Optionally downloads only a specific part of the video
    
    Args:
        url (str): URL of the video/audio to download (e.g., YouTube link)
        output_dir (str): Directory to save the downloaded file (default: "download/")
        start_time (float, optional): Start time in seconds (e.g., 30.5 for 30.5 seconds)
        duration (float, optional): Duration in seconds (e.g., 60 for 1 minute)
        
    Example:
        download_audio("https://youtu.be/example", "my_audio/", 30, 60)
        # Downloads 1 minute starting from 30 seconds
    """
    # Step 1: Set up download options
    # These options tell yt-dlp how to download and process the audio
    ydl_options = {
        "format": "bestaudio/best",  # Get the best audio quality available
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",  # File naming pattern
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",  # Extract audio from video
                "preferredcodec": "mp3",      # Convert to MP3 format
                "preferredquality": "192",    # Audio quality: 192 kbps
                # Quality options: 128kbps (low), 192kbps (medium), 320kbps (high)
            }
        ],
    }
    
    # Step 2: Handle partial downloads (if start_time and duration are specified)
    # This allows you to download only a specific part of a video
    if start_time is not None and duration is not None:
        ydl_options["external_downloader"] = "ffmpeg"  # Use FFmpeg for partial downloads
        ffmpeg_args = []  # Arguments to pass to FFmpeg
        
        if start_time is not None:
            ffmpeg_args.extend(["-ss", str(start_time)])  # Start time
        if duration is not None:
            ffmpeg_args.extend(["-t", str(duration)])     # Duration
            
        ydl_options["external_downloader_args"] = {"ffmpeg": ffmpeg_args}

    # Step 3: Download the audio
    # Create a YouTube downloader with our options and download the file
    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download([url])  # Download the URL we specified
        
    print(f"✓ Audio downloaded successfully to {output_dir}")


def download_video(url, output_dir="download/", start_time=None, duration=None):
    """
    Download video from a URL in MP4 format.
    
    This function downloads video from YouTube or other video sites and saves
    it as an MP4 file. It's perfect for getting video files to experiment with
    in your multimedia processing projects.
    
    What it does:
    1. Takes a YouTube URL (or other video URL)
    2. Downloads the best quality video and audio available
    3. Merges them into a single MP4 file
    4. Saves it to your specified directory
    5. Optionally downloads only a specific part of the video
    
    Args:
        url (str): URL of the video to download (e.g., YouTube link)
        output_dir (str): Directory to save the downloaded file (default: "download/")
        start_time (float, optional): Start time in seconds (e.g., 30.5 for 30.5 seconds)
        duration (float, optional): Duration in seconds (e.g., 60 for 1 minute)
        
    Example:
        download_video("https://youtu.be/example", "my_videos/", 30, 60)
        # Downloads 1 minute of video starting from 30 seconds
    """
    # Step 1: Set up download options
    # These options tell yt-dlp how to download and process the video
    ydl_options = {
        "format": "bestvideo+bestaudio/best",  # Get best video+audio, fallback to best single format
        "outtmpl": f"{output_dir}/%(title)s.%(ext)s",  # File naming pattern
        "merge_output_format": "mp4",  # Automatically merge video and audio into MP4
    }
    
    # Step 2: Handle partial downloads (if start_time and duration are specified)
    # This allows you to download only a specific part of a video
    if start_time is not None and duration is not None:
        ydl_options["external_downloader"] = "ffmpeg"  # Use FFmpeg for partial downloads
        ffmpeg_args = []  # Arguments to pass to FFmpeg
        
        if start_time is not None:
            ffmpeg_args.extend(["-ss", str(start_time)])  # Start time
        if duration is not None:
            ffmpeg_args.extend(["-t", str(duration)])     # Duration
            
        ydl_options["external_downloader_args"] = {"ffmpeg": ffmpeg_args}

    # Step 3: Download the video
    # Create a YouTube downloader with our options and download the file
    with yt_dlp.YoutubeDL(ydl_options) as ydl:
        ydl.download([url])  # Download the URL we specified
        
    print(f"✓ Video downloaded successfully to {output_dir}")

