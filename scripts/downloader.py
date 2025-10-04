"""
DOWNLOADER MODULE
-----------------
Functions for downloading audio and video content from online sources.
"""

import yt_dlp


def download_audio(url, output_dir="download/", start_time=None, duration=None):
    """
    Download audio from a URL and convert to MP3.
    
    Args:
        url (str): URL of the video/audio to download
        output_dir (str): Directory to save the downloaded file
        start_time (float, optional): Start time in seconds
        duration (float, optional): Duration in seconds
    """
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
    print(f"✓ Audio downloaded successfully to {output_dir}")


def download_video(url, output_dir="download/", start_time=None, duration=None):
    """
    Download video from a URL in MP4 format.
    
    Args:
        url (str): URL of the video to download
        output_dir (str): Directory to save the downloaded file
        start_time (float, optional): Start time in seconds
        duration (float, optional): Duration in seconds
    """
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
    print(f"✓ Video downloaded successfully to {output_dir}")

