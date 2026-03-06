"""
AI Drone Rescue Monitoring System — Flask Backend
Supports: Local (webcam + YOLOv8), Desktop (PyWebView), Web (demo mode for hosting)
"""

import os
import sys
import time
import base64
import datetime
import threading
import random
from flask import Flask, Response, render_template, jsonify, send_from_directory

# ─── Mode Detection ───────────────────────────────────────────────────────────
# Set WEB_MODE=true in environment for Render/Railway hosting (no webcam)
WEB_MODE = os.environ.get("WEB_MODE", "false").lower() == "true"

detector = None

if not WEB_MODE:
    try:
        from pose_detection import PoseDetector
        detector = PoseDetector()
        detector.start()
        print("[APP] PoseDetector started — LOCAL mode (webcam + YOLOv8).")
    except Exception as e:
        print(f"[WARN] PoseDetector failed: {e}")
        print("[APP] Switching to WEB_MODE demo simulation.")
        WEB_MODE = True
else:
    print("[APP] WEB_MODE active — running demo simulation (no webcam needed).")

# ─── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── Telemetry simulation (always running) ────────────────────────────────────
_telem = {"altitude": 45.2, "speed": 12.4, "battery": 87.0, "signal": 94.0, "heading": 0}

def _telem_loop():
    while True:
        _telem["altitude"] = round(_telem["altitude"] + random.uniform(-0.5, 0.5), 1)
        _telem["speed"]    = round(max(0, _telem["speed"] + random.uniform(-0.3, 0.3)), 1)
        _telem["battery"]  = max(0, _telem["battery"] - random.uniform(0, 0.05))
        _telem["signal"]   = min(100, max(60, _telem["signal"] + random.uniform(-1, 1)))
        _telem["heading"]  = (_telem["heading"] + random.randint(-5, 5)) % 360
        time.sleep(1)

threading.Thread(target=_telem_loop, daemon=True).start()

# ─── Demo simulation (WEB_MODE) ───────────────────────────────────────────────
DEMO_CYCLE = [
    {"pose": "standing", "status": "NORMAL",    "persons": 1, "standing": 1, "sitting": 0, "lying": 0, "emergency": False},
    {"pose": "standing", "status": "NORMAL",    "persons": 2, "standing": 2, "sitting": 0, "lying": 0, "emergency": False},
    {"pose": "sitting",  "status": "WARNING",   "persons": 1, "standing": 0, "sitting": 1, "lying": 0, "emergency": False},
    {"pose": "standing", "status": "NORMAL",    "persons": 1, "standing": 1, "sitting": 0, "lying": 0, "emergency": False},
    {"pose": "lying",    "status": "EMERGENCY", "persons": 1, "standing": 0, "sitting": 0, "lying": 1, "emergency": True},
    {"pose": "lying",    "status": "EMERGENCY", "persons": 1, "standing": 0, "sitting": 0, "lying": 1, "emergency": True},
    {"pose": "standing", "status": "NORMAL",    "persons": 1, "standing": 1, "sitting": 0, "lying": 0, "emergency": False},
    {"pose": "none",     "status": "SCANNING",  "persons": 0, "standing": 0, "sitting": 0, "lying": 0, "emergency": False},
]

_demo = {
    "state": dict(DEMO_CYCLE[0]),
    "cycle": 0,
    "total": 0,
    "emergency_sector": None,
    "log": [],
}

def _demo_log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    _demo["log"].insert(0, f"[{ts}] {msg}")
    if len(_demo["log"]) > 50:
        _demo["log"].pop()

_demo_log("System initialized. AI Drone Rescue System ready.")
_demo_log("YOLOv8n-pose model loaded successfully.")
_demo_log("Webcam simulation active — Demo Mode")
_demo_log("Drone online — scanning sector A1")

def _demo_loop():
    while True:
        time.sleep(4)
        step = DEMO_CYCLE[_demo["cycle"] % len(DEMO_CYCLE)]
        _demo["state"] = dict(step)
        _demo["total"] += step["persons"]
        _demo["cycle"] += 1

        if step["status"] == "EMERGENCY":
            sec = random.choice(["B2", "C3", "D1", "A3", "B4", "C2"])
            _demo["emergency_sector"] = sec
            _demo_log("🚨 EMERGENCY — Person lying on ground detected!")
            _demo_log(f"Drone dispatched to sector {sec}")
            _demo_log("Alarm activated — rescue team alerted")
        elif step["status"] == "WARNING":
            _demo["emergency_sector"] = None
            _demo_log("⚠️  Person detected — pose: SITTING")
        elif step["status"] == "NORMAL" and step["persons"] > 0:
            _demo["emergency_sector"] = None
            _demo_log(f"✅ Person detected — pose: STANDING (×{step['persons']})")
        elif step["status"] == "SCANNING":
            _demo["emergency_sector"] = None
            _demo_log(f"🔍 Scanning... no persons in frame")
            _demo_log(f"Drone repositioning to sector {random.choice(['A2','B1','C4','D3'])}")

if WEB_MODE:
    threading.Thread(target=_demo_loop, daemon=True).start()

# ─── Grid pathfinding ─────────────────────────────────────────────────────────
ROWS = ["A","B","C","D"]
COLS = [1,2,3,4]

def sector_to_coords(s):
    return (ROWS.index(s[0]), int(s[1]) - 1)

def coords_to_sector(r, c):
    return f"{ROWS[r]}{COLS[c]}"

def bfs(start, goal):
    from collections import deque
    q = deque([[start]])
    visited = {start}
    while q:
        path = q.popleft()
        if path[-1] == goal:
            return path
        r, c = path[-1]
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < 4 and 0 <= nc < 4 and (nr,nc) not in visited:
                visited.add((nr,nc))
                q.append(path + [(nr,nc)])
    return [start, goal]

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html", web_mode=WEB_MODE)

@app.route("/video_feed")
def video_feed():
    if WEB_MODE:
        return Response(status=204)

    def generate():
        while True:
            frame = detector.get_frame()
            if frame:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.04)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/detections")
def detections():
    if WEB_MODE:
        esec = _demo["emergency_sector"]
        result = {
            **_demo["state"],
            "total_detections":  _demo["total"],
            "emergency_sector":  esec,
            "timestamp":         time.time(),
            "log":               list(_demo["log"]),
            "capture_b64":       None,
            "web_mode":          True,
        }
    else:
        result = detector.get_result()
        esec = result.get("emergency_sector")
        cap = result.get("capture")
        if cap and os.path.exists(cap):
            with open(cap, "rb") as f:
                result["capture_b64"] = base64.b64encode(f.read()).decode()
        else:
            caps = sorted([f for f in os.listdir("captures") if f.endswith(".jpg")])
            result["capture_b64"] = None
            if caps:
                with open(os.path.join("captures", caps[-1]), "rb") as f:
                    result["capture_b64"] = base64.b64encode(f.read()).decode()
        result["web_mode"] = False

    # Telemetry
    result["telemetry"] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in _telem.items()}

    # Drone path
    if esec:
        path = bfs(sector_to_coords("A1"), sector_to_coords(esec))
        result["drone_path"] = [coords_to_sector(r, c) for r, c in path]
    else:
        result["drone_path"] = []

    return jsonify(result)

@app.route("/status")
def status():
    return jsonify({
        "ai_model":  "YOLOv8n-pose" + (" (Demo)" if WEB_MODE else " (Live)"),
        "camera":    "Simulated" if WEB_MODE else "Connected",
        "detection": "Active",
        "network":   "Online",
        "mode":      "WEB_DEMO" if WEB_MODE else "LOCAL_LIVE",
        "uptime":    int(time.time()),
    })

@app.route("/captures/<filename>")
def serve_capture(filename):
    return send_from_directory("captures", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  AI DRONE RESCUE MONITORING SYSTEM")
    print(f"  Mode  : {'WEB DEMO (no webcam)' if WEB_MODE else 'LOCAL LIVE (webcam + YOLOv8)'}")
    print(f"  URL   : http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
