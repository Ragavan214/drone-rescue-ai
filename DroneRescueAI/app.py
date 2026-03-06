"""
AI Drone Rescue Monitoring System — Flask Backend
Supports: Local (webcam + YOLOv8), Web (browser webcam via JS → Flask)
"""

import os
import sys
import time
import base64
import datetime
import threading
import random
import numpy as np
import cv2
from flask import Flask, Response, render_template, jsonify, send_from_directory, request

# ─── Mode Detection ───────────────────────────────────────────────────────────
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
        print("[APP] Switching to WEB_MODE.")
        WEB_MODE = True
else:
    print("[APP] WEB_MODE active — browser webcam mode.")

# ─── Flask ────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ─── YOLOv8 for web frame processing ──────────────────────────────────────────
yolo_model = None
if WEB_MODE:
    try:
        from ultralytics import YOLO
        yolo_model = YOLO("yolov8n-pose.pt")
        print("[APP] YOLOv8n-pose loaded for web frame processing.")
    except Exception as e:
        print(f"[WARN] YOLO load failed: {e}")

# ─── Per-session state (keyed by session_id) ──────────────────────────────────
_sessions = {}
_sessions_lock = threading.Lock()

def get_session(sid):
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = {
                "pose": "none",
                "status": "SCANNING",
                "persons": 0,
                "standing": 0,
                "sitting": 0,
                "lying": 0,
                "emergency": False,
                "total_detections": 0,
                "total_standing": 0,
                "total_sitting": 0,
                "total_lying": 0,
                "emergency_sector": None,
                "log": [],
                "capture_b64": None,
                "last_update": time.time(),
            }
        return _sessions[sid]

def session_log(sid, msg):
    s = get_session(sid)
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    s["log"].insert(0, f"[{ts}] {msg}")
    if len(s["log"]) > 50:
        s["log"].pop()

# ─── Telemetry simulation ─────────────────────────────────────────────────────
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

# ─── Pose classification from keypoints ───────────────────────────────────────
def classify_pose(keypoints):
    """Classify pose as standing/sitting/lying from YOLOv8 keypoints."""
    try:
        kp = keypoints
        # nose=0, l_shoulder=5, r_shoulder=6, l_hip=11, r_hip=12, l_knee=13, r_knee=14, l_ankle=15, r_ankle=16
        def valid(idx):
            return kp[idx][2] > 0.3  # confidence threshold

        nose_y      = kp[0][1]  if valid(0)  else None
        l_shoulder  = kp[5][1]  if valid(5)  else None
        r_shoulder  = kp[6][1]  if valid(6)  else None
        l_hip       = kp[11][1] if valid(11) else None
        r_hip       = kp[12][1] if valid(12) else None
        l_ankle     = kp[15][1] if valid(15) else None
        r_ankle     = kp[16][1] if valid(16) else None

        shoulder_y = None
        if l_shoulder and r_shoulder:
            shoulder_y = (l_shoulder + r_shoulder) / 2
        elif l_shoulder:
            shoulder_y = l_shoulder
        elif r_shoulder:
            shoulder_y = r_shoulder

        hip_y = None
        if l_hip and r_hip:
            hip_y = (l_hip + r_hip) / 2
        elif l_hip:
            hip_y = l_hip
        elif r_hip:
            hip_y = r_hip

        ankle_y = None
        if l_ankle and r_ankle:
            ankle_y = (l_ankle + r_ankle) / 2
        elif l_ankle:
            ankle_y = l_ankle
        elif r_ankle:
            ankle_y = r_ankle

        if shoulder_y is None or hip_y is None:
            return "standing"

        vertical_span = abs(hip_y - (nose_y or shoulder_y))
        frame_height = 480  # approximate

        # Lying: body is mostly horizontal (small vertical span)
        if vertical_span < frame_height * 0.15:
            return "lying"

        # Sitting: hips visible but ankles not far below hips, or legs bent
        if ankle_y is None:
            return "sitting"

        leg_length = abs(ankle_y - hip_y)
        torso_length = abs(hip_y - shoulder_y)

        if leg_length < torso_length * 0.8:
            return "sitting"

        return "standing"

    except Exception:
        return "standing"

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

@app.route("/process_frame", methods=["POST"])
def process_frame():
    """Receive a base64 frame from browser webcam, run YOLOv8, return results."""
    if not WEB_MODE:
        return jsonify({"error": "not in web mode"}), 400

    data = request.get_json()
    if not data or "frame" not in data:
        return jsonify({"error": "no frame"}), 400

    sid = data.get("session_id", "default")
    session = get_session(sid)

    try:
        # Decode base64 image
        img_data = data["frame"].split(",")[1] if "," in data["frame"] else data["frame"]
        img_bytes = base64.b64decode(img_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"error": "invalid frame"}), 400

        result_data = {
            "pose": "none", "status": "SCANNING",
            "persons": 0, "standing": 0, "sitting": 0, "lying": 0,
            "emergency": False, "annotated_frame": None
        }

        if yolo_model is not None:
            results = yolo_model(frame, verbose=False, conf=0.4)
            result = results[0]

            standing = sitting = lying = 0
            emergency = False

            if result.keypoints is not None and len(result.keypoints.data) > 0:
                for kp_data in result.keypoints.data:
                    kp = kp_data.cpu().numpy()
                    pose = classify_pose(kp)
                    if pose == "standing":   standing += 1
                    elif pose == "sitting":  sitting += 1
                    elif pose == "lying":
                        lying += 1
                        emergency = True

            persons = standing + sitting + lying

            # Determine overall status
            if lying > 0:
                status = "EMERGENCY"
                pose_label = "lying"
            elif sitting > 0:
                status = "WARNING"
                pose_label = "sitting"
            elif standing > 0:
                status = "NORMAL"
                pose_label = "standing"
            else:
                status = "SCANNING"
                pose_label = "none"

            # Update session state
            session["pose"]     = pose_label
            session["status"]   = status
            session["persons"]  = persons
            session["standing"] = standing
            session["sitting"]  = sitting
            session["lying"]    = lying
            session["emergency"] = emergency
            session["total_detections"] += persons
            session["total_standing"]   += standing
            session["total_sitting"]    += sitting
            session["total_lying"]      += lying
            session["last_update"]      = time.time()

            # Log events
            if emergency and not session.get("_last_emg"):
                session["_last_emg"] = True
                session["emergency_sector"] = random.choice(["B2","C3","D1","A3","B4","C2"])
                session_log(sid, "🚨 EMERGENCY — Person lying on ground detected!")
                session_log(sid, f"Drone dispatched to sector {session['emergency_sector']}")
            elif not emergency:
                session["_last_emg"] = False
                session["emergency_sector"] = None
                if persons > 0:
                    session_log(sid, f"✅ {persons} person(s) detected — {pose_label.upper()}")

            # Annotate frame and encode back
            annotated = result.plot()
            _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            annotated_b64 = base64.b64encode(buffer).decode()

            result_data = {
                "pose": pose_label,
                "status": status,
                "persons": persons,
                "standing": standing,
                "sitting": sitting,
                "lying": lying,
                "emergency": emergency,
                "annotated_frame": "data:image/jpeg;base64," + annotated_b64,
            }

            # Save emergency capture
            if emergency:
                os.makedirs("captures", exist_ok=True)
                ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                cap_path = os.path.join("captures", f"emergency_{ts}.jpg")
                cv2.imwrite(cap_path, annotated)
                with open(cap_path, "rb") as f:
                    session["capture_b64"] = base64.b64encode(f.read()).decode()

    except Exception as e:
        print(f"[ERROR] process_frame: {e}")
        return jsonify({"error": str(e)}), 500

    return jsonify(result_data)

@app.route("/detections")
def detections():
    if WEB_MODE:
        sid = request.args.get("session_id", "default")
        session = get_session(sid)
        esec = session["emergency_sector"]
        result = {
            "pose":             session["pose"],
            "status":           session["status"],
            "persons":          session["persons"],
            "standing":         session["standing"],
            "sitting":          session["sitting"],
            "lying":            session["lying"],
            "emergency":        session["emergency"],
            "total_detections": session["total_detections"],
            "total_standing":   session["total_standing"],
            "total_sitting":    session["total_sitting"],
            "total_lying":      session["total_lying"],
            "emergency_sector": esec,
            "capture_b64":      session["capture_b64"],
            "log":              list(session["log"]),
            "web_mode":         True,
            "timestamp":        time.time(),
        }
    else:
        result = detector.get_result()
        esec = result.get("emergency_sector")
        cap = result.get("capture")
        if cap and os.path.exists(cap):
            with open(cap, "rb") as f:
                result["capture_b64"] = base64.b64encode(f.read()).decode()
        else:
            result["capture_b64"] = None
        result["web_mode"] = False

    result["telemetry"] = {k: (round(v,1) if isinstance(v, float) else v) for k,v in _telem.items()}

    if result.get("emergency_sector"):
        path = bfs(sector_to_coords("A1"), sector_to_coords(result["emergency_sector"]))
        result["drone_path"] = [coords_to_sector(r,c) for r,c in path]
    else:
        result["drone_path"] = []

    return jsonify(result)

@app.route("/status")
def status():
    return jsonify({
        "ai_model":  "YOLOv8n-pose" + (" (Web)" if WEB_MODE else " (Live)"),
        "camera":    "Browser Webcam" if WEB_MODE else "Connected",
        "detection": "Active",
        "network":   "Online",
        "mode":      "WEB_LIVE" if WEB_MODE else "LOCAL_LIVE",
        "uptime":    int(time.time()),
    })

@app.route("/captures/<filename>")
def serve_capture(filename):
    return send_from_directory("captures", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 60)
    print("  AI DRONE RESCUE MONITORING SYSTEM")
    print(f"  Mode  : {'WEB LIVE (browser webcam)' if WEB_MODE else 'LOCAL LIVE (webcam + YOLOv8)'}")
    print(f"  URL   : http://0.0.0.0:{port}")
    print("=" * 60)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
