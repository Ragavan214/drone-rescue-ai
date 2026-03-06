"""
AI Drone Rescue — Desktop EXE Builder
Run: python build.py
"""

import os, sys, subprocess, shutil

def build():
    print("=" * 60)
    print("  AI DRONE RESCUE — Desktop EXE Builder")
    print("=" * 60)

    # 1. Install packages
    print("\n[1/4] Installing packages...")
    pkgs = ["pywebview", "pyinstaller", "flask", "ultralytics", "opencv-python", "numpy"]
    for p in pkgs:
        print(f"      -> {p}")
        subprocess.run([sys.executable, "-m", "pip", "install", p, "--quiet"], check=False)
    print("      Done.")

    # 2. Clean old builds
    print("\n[2/4] Cleaning old builds...")
    for d in ["build", "dist", "__pycache__"]:
        if os.path.exists(d):
            shutil.rmtree(d)
    for f in os.listdir("."):
        if f.endswith(".spec"):
            os.remove(f)
    print("      Done.")

    # 3. PyInstaller
    print("\n[3/4] Building .exe (3-5 mins)...\n")
    data_sep = ";" if sys.platform == "win32" else ":"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--onedir", "--windowed",
        "--name", "DroneRescueAI",
        "--add-data", f"templates{data_sep}templates",
        "--add-data", f"static{data_sep}static",
        "--add-data", f"captures{data_sep}captures",
        "--hidden-import", "flask",
        "--hidden-import", "webview",
        "--hidden-import", "webview.platforms.winforms",
        "--hidden-import", "clr",
        "--hidden-import", "ultralytics",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "engineio.async_drivers.threading",
        "--hidden-import", "pkg_resources.py2_compat",
        "--collect-all", "ultralytics",
        "--collect-all", "webview",
        "desktop_app.py",
    ]

    if os.path.exists("icon.ico"):
        cmd += ["--icon", "icon.ico"]

    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("\n[ERROR] Build failed. See errors above.")
        return

    # 4. Copy model + finish
    print("\n[4/4] Finalising...")
    dest = os.path.join("dist", "DroneRescueAI")
    if os.path.exists("yolov8n-pose.pt"):
        shutil.copy("yolov8n-pose.pt", dest)
        print("      Copied YOLOv8 model to dist folder.")

    print("\n" + "=" * 60)
    print("  SUCCESS!")
    print()
    print("  Desktop app is at:")
    print("  dist\\DroneRescueAI\\DroneRescueAI.exe")
    print()
    print("  To share: zip dist\\DroneRescueAI\\ folder")
    print("  Recipient just extracts and double-clicks!")
    print("=" * 60)

if __name__ == "__main__":
    build()
