#!/bin/bash

echo ""
echo "============================================================"
echo "  AI DRONE RESCUE MONITORING SYSTEM"
echo "  Automatic Setup and Launcher"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] Python 3 is not installed."
    echo ""
    echo "Install it from: https://www.python.org/downloads/"
    echo "Or on Mac with Homebrew: brew install python3"
    exit 1
fi

echo "[OK] Python found: $(python3 --version)"
echo ""

# Install packages
echo "[INFO] Installing required packages (first time may take 2-5 minutes)..."
pip3 install -r requirements.txt --quiet

if [ $? -ne 0 ]; then
    echo "[ERROR] Package installation failed."
    echo "Try: sudo pip3 install -r requirements.txt"
    exit 1
fi

echo ""
echo "[OK] All packages installed!"
echo ""
echo "============================================================"
echo "  Launching AI Drone Rescue Desktop App..."
echo "============================================================"
echo ""

python3 desktop_app.py
