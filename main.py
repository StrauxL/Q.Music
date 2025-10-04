"""
MULTIMEDIA PROCESSING PLAYGROUND - MAIN SCRIPT
-----------------------------------------------
A collection of tools for manipulating audio, video, and images.
This script orchestrates the various multimedia processing functions.
"""

import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts import config
from scripts.downloader import download_audio, download_video
from scripts.audio_info import basic_audio_info
from scripts.audio_effects import create_audio_mashup
from scripts.voice_effects import create_voice_changer, create_nightcore_effect
from scripts import library as lib



'''''''''''''''''''''''''''''
DOWNLOAD MODE
'''''''''''''''''''''''''''''
download_video(config.satisfya_eng, output_dir=config.DOWNLOAD_DIR)
download_audio(config.dhivara_malayalam, output_dir=config.DOWNLOAD_DIR)
# lib.Audio(filename=f"{config.OUTPUT_DIR}/audio_mashup.wav")

'''''''''''''''''''''''''''''
LAB MODE
'''''''''''''''''''''''''''''
y, sr = basic_audio_info(config.AUDIO_PATH)
    
mashup = create_audio_mashup(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, y=y, sr=sr)

voice_effects = create_voice_changer(config.AUDIO_PATH, output_dir=config.OUTPUT_DIR, y=y, sr=sr)

demo_studio_effects()
demo_synthesis()
demo_mixing()
demo_analysis()
run_all_demos()
