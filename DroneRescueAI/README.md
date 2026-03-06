# 🚁 AI Drone Rescue Monitoring System

> **Real-time human pose detection for emergency rescue operations using YOLOv8 AI**

A full-stack AI application that uses computer vision to detect people via drone camera feed, classify their body pose (standing / sitting / lying), and automatically trigger emergency alerts when someone is detected lying on the ground — dispatching a simulated rescue drone to the exact grid sector.

---

## 🌐 Live Demo

> Deploy to Render or Railway — anyone can access via browser with zero installation.

**[▶ Open Live Demo](https://your-app-name.onrender.com)**  ← *replace after deploying*

---

## 📸 Features

| Feature | Description |
|---------|-------------|
| 🤖 YOLOv8 Pose AI | Real-time 17-keypoint skeleton detection |
| 🔴 Emergency Detection | Auto-alerts when person is lying (injured) |
| 🗺️ Grid Sector Map | 4×4 grid with BFS pathfinding for drone routing |
| 📊 Live Telemetry | Altitude, speed, battery, signal, heading |
| 📋 Mission Log | Real-time event log with timestamps |
| 💻 Dual Mode | Live webcam (local) or animated demo (web hosted) |
| 🖥️ Desktop App | Standalone .exe — no browser needed |

---

## 🚀 Quick Start

### Option A — Run Locally (with webcam + live AI)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run
python app.py

# 3. Open browser
# http://localhost:5000
```

### Option B — Desktop App (native window)

```bash
pip install -r requirements.txt
python desktop_app.py
```

### Option C — Demo Mode (no webcam needed)

```bash
pip install flask
WEB_MODE=true python app.py
# Windows: set WEB_MODE=true && python app.py
```

---

## ☁️ Deploy to Render (Free Hosting)

### Step 1 — Push to GitHub
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/drone-rescue-ai.git
git push -u origin main
```

### Step 2 — Deploy on Render
1. Go to **https://render.com** → Sign up free
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repo
4. Fill in the settings:

| Setting | Value |
|---------|-------|
| **Name** | `drone-rescue-ai` |
| **Runtime** | `Python 3` |
| **Build Command** | `pip install -r requirements-web.txt` |
| **Start Command** | `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 4` |

5. Add **Environment Variable**:
   - Key: `WEB_MODE`  
   - Value: `true`

6. Click **"Create Web Service"** → Wait 2-3 minutes
7. Your app is live at `https://drone-rescue-ai.onrender.com` 🎉

---

## ☁️ Deploy to Railway (Alternative)

1. Go to **https://railway.app** → Sign up free
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repo — Railway auto-detects `railway.toml`
4. Done! Your URL appears in the dashboard.

---

## 🖥️ Build Standalone .EXE (Windows)

To create a desktop app that works without Python installed:

```bash
python build.py
```

Wait 3–5 minutes. Output:
```
dist\DroneRescueAI\DroneRescueAI.exe
```

Zip the `dist\DroneRescueAI\` folder and share — anyone can double-click and run!

---

## 📁 Project Structure

```
DroneRescueAI/
│
├── app.py                  ← Flask backend (LOCAL + WEB modes)
├── pose_detection.py       ← YOLOv8 AI engine + pose classifier
├── desktop_app.py          ← PyWebView desktop launcher
├── build.py                ← Builds standalone .exe
│
├── templates/
│   └── index.html          ← Full HUD dashboard (HTML/CSS/JS)
│
├── static/                 ← Static assets
├── captures/               ← Emergency screenshots saved here
│
├── requirements.txt        ← Full deps (local / desktop)
├── requirements-web.txt    ← Minimal deps (hosting — Flask only)
├── Procfile                ← For Railway / Heroku
├── render.yaml             ← For Render auto-deploy
├── railway.toml            ← For Railway auto-deploy
│
├── LAUNCH_WINDOWS.bat      ← One-click Windows launcher
└── LAUNCH_MAC_LINUX.sh     ← One-click Mac/Linux launcher
```

---

## 🧠 How It Works

```
Webcam Frame
    │
    ▼
YOLOv8n-pose  ──►  17 Keypoints (COCO format)
    │
    ▼
Pose Classifier
    ├── Standing  →  NORMAL   (green)
    ├── Sitting   →  WARNING  (orange)
    └── Lying     →  EMERGENCY (red) ──► Drone dispatched
                                              │
                                              ▼
                                      BFS Pathfinding
                                      on 4×4 grid map
```

**Pose Classification Rules:**
- **Lying**: Head Y ≈ Hip Y (horizontal body, torso angle > 55°)
- **Sitting**: Torso angle 15–55°, knees near hip level
- **Standing**: Torso vertical, full body elongated top-to-bottom

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| AI Model | YOLOv8n-pose (Ultralytics) |
| Computer Vision | OpenCV |
| Backend | Python + Flask |
| Frontend | HTML5 / CSS3 / Vanilla JS |
| Desktop | PyWebView |
| Packaging | PyInstaller |
| Hosting | Render / Railway |

---

## ❓ Troubleshooting

**Webcam not detected**
- Close other apps using the camera (Zoom, Teams, etc.)
- On the .exe, try running as Administrator

**Black window on launch**
- Wait 5–10 seconds for Flask to initialise
- Try resizing the window

**"Module not found" error**
```bash
pip install -r requirements.txt
```

**App works but no AI detections**
- Ensure good lighting
- Stand ~1–3 metres from camera
- The model detects full-body poses best

---

## 📄 License

MIT License — free to use, modify, and distribute for educational purposes.

---

*Built with ❤️ using YOLOv8 · Flask · OpenCV · PyWebView*
