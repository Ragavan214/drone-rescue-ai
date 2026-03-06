"""
AI Drone Rescue Monitoring System — Desktop App
Uses PyWebView to open the Flask app in a native desktop window.
"""

import threading
import time
import sys
import os
import webview

# ── Make sure relative imports work when bundled with PyInstaller ──────────
if getattr(sys, 'frozen', False):
    # Running as compiled .exe
    BASE_DIR = sys._MEIPASS
    os.chdir(BASE_DIR)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

# ── Import and start Flask app ─────────────────────────────────────────────
from app import app, detector

def start_flask():
    """Run Flask in a background thread."""
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False, threaded=True)

def main():
    print("=" * 55)
    print("  AI DRONE RESCUE MONITORING SYSTEM")
    print("  Desktop Application — Starting...")
    print("=" * 55)

    # Start Flask in background
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()

    # Wait for Flask to be ready
    print("[BOOT] Starting Flask server...")
    time.sleep(2)
    print("[BOOT] Opening desktop window...")

    # Create PyWebView desktop window
    window = webview.create_window(
        title="AI Drone Rescue Monitoring System",
        url="http://127.0.0.1:5000",
        width=1400,
        height=860,
        min_size=(1000, 650),
        resizable=True,
        fullscreen=False,
        background_color="#020a14",
        text_select=False,
    )

    # Start the GUI (blocks until window is closed)
    webview.start(debug=False)

    # Cleanup on close
    print("[EXIT] Window closed. Shutting down...")
    detector.stop()
    sys.exit(0)


if __name__ == "__main__":
    main()
