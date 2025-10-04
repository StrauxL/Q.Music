#!/bin/bash
# Activation script for Q.Music venv environment
# This ensures ffmpeg and all Python packages work correctly

cd "$(dirname "$0")"
source venv/bin/activate

echo "✓ Q.Music venv activated!"
echo "  Python: $(python --version)"
echo "  yt-dlp: $(yt-dlp --version)"
echo "  ffmpeg: $(ffmpeg -version | head -1)"
echo ""
echo "You can now run your Python scripts or start Jupyter:"
echo "  python garden.py"
echo "  jupyter notebook"
echo ""
echo "To deactivate, type: deactivate"

# Keep shell active
exec bash






