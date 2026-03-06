@echo off
title AI Drone Rescue — Setup & Launch
color 0A

echo.
echo  ============================================================
echo    AI DRONE RESCUE MONITORING SYSTEM
echo    Automatic Setup and Launcher
echo  ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python is not installed or not in PATH.
    echo.
    echo  Please install Python 3.9 or newer from:
    echo  https://www.python.org/downloads/
    echo.
    echo  IMPORTANT: During install, tick "Add Python to PATH"
    echo.
    pause
    exit /b 1
)

echo  [OK] Python found.
echo.

:: Check pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] pip not found. Please reinstall Python with pip enabled.
    pause
    exit /b 1
)

echo  [INFO] Installing required packages (this may take 2-5 minutes)...
echo         (YOLOv8 model will also be downloaded on first run ~6MB)
echo.

pip install -r requirements.txt --quiet

if errorlevel 1 (
    echo.
    echo  [ERROR] Package installation failed.
    echo  Try running this file as Administrator.
    pause
    exit /b 1
)

echo.
echo  [OK] All packages installed!
echo.
echo  ============================================================
echo    Launching AI Drone Rescue Desktop App...
echo  ============================================================
echo.

python desktop_app.py

pause
