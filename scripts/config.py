"""
CONFIGURATION MODULE
--------------------
Contains all configuration constants and URL references for the multimedia processing playground.

This module stores all the URLs and file paths used throughout the project.
It's like a central place where all the important settings are kept.

Key Concepts for Beginners:
- URL: Web address that points to a video or audio file
- File Path: Location of a file on your computer
- Configuration: Settings that control how the program behaves
- Constants: Values that don't change during program execution

Author: Q.Wave Team
"""

# =============================================================================
# AUDIO/VIDEO URLs
# =============================================================================
# These are YouTube URLs for various songs in different languages
# You can use these URLs with the downloader functions to get audio/video files

# Apple Struggle - A popular song
apple_struggle = "https://youtu.be/vN4U5FqrOdQ"

# Firestorm - Available in multiple languages
firestorm_hindi = "https://youtu.be/701FwxtMdpw"    # Hindi version
firestorm_telegu = "https://youtu.be/FbXOsVByKmk"   # Telugu version

# Satisfya - Available in multiple languages
satisfya_eng = "https://youtu.be/-_zKab2r3fw?list=PL5z9QulhxYohPKSJoW33ogvS6mM0RXknG"  # English version
satisfya_hindi = "https://youtu.be/pfVODjDBFxU"     # Hindi version

# Mayabini - Available in multiple languages
mayabini_hindi = "https://youtu.be/jX9KIpErgQg"     # Hindi version
mayabini_english = "https://youtu.be/gWhGA4JSGEk"   # English version
mayabini_assamese = "https://youtu.be/xreNppGG1lM"  # Assamese version

# Dhivara - Available in multiple languages (from Baahubali movie)
dhivara_eng = "https://youtu.be/696LdKpzpUA"        # English version
dhivara_telegu = "https://youtu.be/6jpcUd5A5Jw"     # Telugu version
dhivara_hindi = "https://youtu.be/_2clW8Zxq88"      # Hindi version
dhivara_tamil = "https://youtu.be/YAMNFIj9NwU"      # Tamil version
dhivara_malayalam = "https://youtu.be/aSMT7c7Q3Rc"  # Malayalam version

# Aikon Baikon - Available in multiple languages
AikonBaikon_eng = "https://youtu.be/rx0vmM93QDs"    # English version
AikonBaikon_assamese = "https://youtu.be/sGfmDgiYAL0"  # Assamese version

# Shadowborn - Portuguese version
Shadowborn_portuguese = "https://youtu.be/yhg2SZp7U0k?list=RDyhg2SZp7U0k"

# =============================================================================
# FILE PATH CONFIGURATION
# =============================================================================
# These are the default paths where files are stored and saved

# Default audio file to use for testing and examples
AUDIO_PATH = "input/FirestormOG.mp3"  # Path to the default audio file

# Directory where processed audio files are saved
OUTPUT_DIR = "output"  # All generated audio files go here

# Directory where downloaded files are stored
DOWNLOAD_DIR = "download/"  # All downloaded audio/video files go here

