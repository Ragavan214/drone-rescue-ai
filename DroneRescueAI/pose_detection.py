"""
Pose Detection Module using YOLOv8 Pose Model.
Classifies human poses as: standing, sitting, or lying (emergency).
"""

import cv2
import numpy as np
import time
import datetime
import os
import threading
from collections import deque

# YOLOv8 keypoint indices (COCO format)
KP_NOSE        = 0
KP_LEFT_EYE    = 1
KP_RIGHT_EYE   = 2
KP_LEFT_EAR    = 3
KP_RIGHT_EAR   = 4
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_ELBOW  = 7
KP_RIGHT_ELBOW = 8
KP_LEFT_WRIST  = 9
KP_RIGHT_WRIST = 10
KP_LEFT_HIP    = 11
KP_RIGHT_HIP   = 12
KP_LEFT_KNEE   = 13
KP_RIGHT_KNEE  = 14
KP_LEFT_ANKLE  = 15
KP_RIGHT_ANKLE = 16

try:
    from ultralytics import YOLO
    print("[YOLO] Loading YOLOv8 pose model...")
    model = YOLO("yolov8n-pose.pt")
    print("[YOLO] Model loaded successfully.")
    YOLO_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to load YOLOv8 model: {e}")
    print("[ERROR] Please install ultralytics: pip install ultralytics")
    YOLO_AVAILABLE = False
    raise RuntimeError(f"YOLOv8 model failed to load: {e}")


class PoseDetector:
    def __init__(self):
        self.model = model
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_result = {
            "pose": "none",
            "status": "NORMAL",
            "persons": 0,
            "standing": 0,
            "sitting": 0,
            "lying": 0,
            "total_detections": 0,
            "emergency": False,
            "emergency_sector": None,
            "timestamp": time.time(),
        }
        self.stats = {"standing": 0, "sitting": 0, "lying": 0, "total": 0}
        self.cap = None
        self.running = False
        self.emergency_active = False
        self.last_emergency_time = 0
        self.emergency_cooldown = 5  # seconds
        self.captures_dir = "captures"
        os.makedirs(self.captures_dir, exist_ok=True)

        # Mission log (max 50 entries)
        self.mission_log = deque(maxlen=50)
        self._log(f"System initialized. YOLOv8 pose model ready.")

    def _log(self, message):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        entry = f"[{ts}] {message}"
        self.mission_log.appendleft(entry)
        print(entry)

    def get_keypoint(self, kps, idx, conf_threshold=0.3):
        """Extract a keypoint if confidence is above threshold."""
        if idx >= len(kps):
            return None
        kp = kps[idx]
        if len(kp) >= 3 and kp[2] >= conf_threshold:
            return (float(kp[0]), float(kp[1]))
        elif len(kp) == 2:
            return (float(kp[0]), float(kp[1]))
        return None

    def midpoint(self, a, b):
        if a is None or b is None:
            return None
        return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)

    def classify_pose(self, keypoints):
        """
        Classify pose based on skeleton keypoints.

        Rules:
        - LYING:    body is horizontal (head y ≈ hip y, relative to frame height)
        - SITTING:  torso is roughly vertical but hips are raised, knees bent
        - STANDING: torso is vertical, body elongated top to bottom
        """
        kps = keypoints

        head   = self.get_keypoint(kps, KP_NOSE)
        l_sho  = self.get_keypoint(kps, KP_LEFT_SHOULDER)
        r_sho  = self.get_keypoint(kps, KP_RIGHT_SHOULDER)
        l_hip  = self.get_keypoint(kps, KP_LEFT_HIP)
        r_hip  = self.get_keypoint(kps, KP_RIGHT_HIP)
        l_knee = self.get_keypoint(kps, KP_LEFT_KNEE)
        r_knee = self.get_keypoint(kps, KP_RIGHT_KNEE)
        l_ank  = self.get_keypoint(kps, KP_LEFT_ANKLE)
        r_ank  = self.get_keypoint(kps, KP_RIGHT_ANKLE)

        shoulder = self.midpoint(l_sho, r_sho)
        hip      = self.midpoint(l_hip, r_hip)
        knee     = self.midpoint(l_knee, r_knee)

        if head is None or shoulder is None or hip is None:
            return "unknown"

        # Vector from shoulder to hip
        torso_dx = hip[0] - shoulder[0]
        torso_dy = hip[1] - shoulder[1]
        torso_len = max(np.sqrt(torso_dx**2 + torso_dy**2), 1)

        # Angle of torso from vertical (0° = perfectly vertical)
        torso_angle = abs(np.degrees(np.arctan2(abs(torso_dx), abs(torso_dy))))

        # Head to hip vertical vs horizontal distance
        head_hip_dx = abs(head[0] - hip[0])
        head_hip_dy = abs(head[1] - hip[1])

        # LYING: head and hip at similar Y level (horizontal body)
        if head_hip_dy < head_hip_dx * 0.6 or torso_angle > 55:
            return "lying"

        # SITTING: torso angle moderate, knees present and elevated
        if knee is not None and torso_angle > 15:
            knee_above_hip = knee[1] < hip[1] + 30  # knee near hip height
            if knee_above_hip or torso_angle > 30:
                return "sitting"

        # Default: STANDING
        return "standing"

    def process_frame(self, frame):
        """Run YOLO inference and classify all detected persons."""
        try:
            results = self.model(frame, verbose=False, conf=0.4)
        except Exception as e:
            self._log(f"Inference error: {e}")
            return frame, []

        persons = []
        annotated = frame.copy()
        
        if not results or results[0].keypoints is None:
            return annotated, persons

        result = results[0]
        
        # Draw skeleton on frame
        annotated = result.plot()

        # Get keypoints for each detected person
        try:
            kps_data = result.keypoints.data.cpu().numpy()
        except Exception:
            return annotated, persons

        for person_kps in kps_data:
            pose = self.classify_pose(person_kps)
            persons.append(pose)

            # Draw pose label
            try:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    if i < len(persons):
                        x1, y1 = int(box[0]), int(box[1])
                        lbl = persons[i].upper()
                        color = (0, 255, 0) if lbl == "STANDING" else \
                                (0, 165, 255) if lbl == "SITTING" else \
                                (0, 0, 255)
                        cv2.rectangle(annotated, (x1, y1 - 30), (x1 + 120, y1), color, -1)
                        cv2.putText(annotated, lbl, (x1 + 5, y1 - 8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            except Exception:
                pass

        return annotated, persons

    def save_emergency_capture(self, frame):
        """Save a JPEG of the emergency frame."""
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.captures_dir, f"emergency_{ts}.jpg")
        cv2.imwrite(path, frame)
        self._log(f"Emergency capture saved: {path}")
        return path

    def get_random_sector(self):
        import random
        rows = ["A", "B", "C", "D"]
        cols = [1, 2, 3, 4]
        return f"{random.choice(rows)}{random.choice(cols)}"

    def camera_loop(self):
        """Main camera capture + detection loop."""
        self._log("Opening webcam...")
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self._log("Webcam not available, trying index 1...")
            self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self._log("[ERROR] Could not open any webcam.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._log("Webcam connected. Starting pose detection loop.")
        self._log("Drone scanning sector A1")

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            annotated, persons = self.process_frame(frame)

            # Compute per-pose counts
            n_standing = persons.count("standing")
            n_sitting  = persons.count("sitting")
            n_lying    = persons.count("lying")
            n_persons  = len(persons)

            self.stats["standing"] += n_standing
            self.stats["sitting"]  += n_sitting
            self.stats["lying"]    += n_lying
            self.stats["total"]    += n_persons

            # Determine overall status
            emergency = n_lying > 0
            if n_lying > 0:
                status = "EMERGENCY"
                pose_label = "lying"
            elif n_sitting > 0:
                status = "WARNING"
                pose_label = "sitting"
            elif n_standing > 0:
                status = "NORMAL"
                pose_label = "standing"
            else:
                status = "SCANNING"
                pose_label = "none"

            # Log detections
            if n_persons > 0 and not self.emergency_active:
                self._log(f"Person detected — pose: {pose_label.upper()}")

            # Emergency handling
            emergency_sector = None
            capture_path = None
            now = time.time()

            if emergency and (now - self.last_emergency_time > self.emergency_cooldown):
                self.emergency_active = True
                self.last_emergency_time = now
                emergency_sector = self.get_random_sector()
                capture_path = self.save_emergency_capture(frame)
                self._log("🚨 EMERGENCY DETECTED — Person lying on ground!")
                self._log("Alarm activated")
                self._log(f"Drone dispatched to sector {emergency_sector}")
            elif not emergency:
                self.emergency_active = False

            # Encode frame to JPEG
            _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])

            with self.lock:
                self.latest_frame = buf.tobytes()
                self.latest_result = {
                    "pose": pose_label,
                    "status": status,
                    "persons": n_persons,
                    "standing": n_standing,
                    "sitting": n_sitting,
                    "lying": n_lying,
                    "total_detections": self.stats["total"],
                    "total_standing": self.stats["standing"],
                    "total_sitting": self.stats["sitting"],
                    "total_lying": self.stats["lying"],
                    "emergency": emergency,
                    "emergency_sector": emergency_sector if emergency else None,
                    "timestamp": now,
                    "capture": capture_path,
                    "log": list(self.mission_log),
                }

            time.sleep(0.03)  # ~30 fps

        if self.cap:
            self.cap.release()

    def start(self):
        self.running = True
        t = threading.Thread(target=self.camera_loop, daemon=True)
        t.start()

    def stop(self):
        self.running = False

    def get_frame(self):
        with self.lock:
            return self.latest_frame

    def get_result(self):
        with self.lock:
            return dict(self.latest_result)
