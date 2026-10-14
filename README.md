# Sentinel-AI: AI-Powered Exam Proctoring & Integrity Monitor

An enterprise-grade, privacy-first **Exam Proctoring & Integrity Monitoring** web application that runs **100% client-side in the candidate's browser** via hardware-accelerated **WebAssembly and WebGL** using Google's `@mediapipe/tasks-vision`.

![Proctoring Dashboard Overview](https://img.shields.io/badge/Computer%20Vision-WebAssembly%20%2F%20WebGL-00f0ff?style=for-the-badge)
![MediaPipe](https://img.shields.io/badge/Google-MediaPipe%20Vision%20v0.10.14-10b981?style=for-the-badge)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20(Optional)-38bdf8?style=for-the-badge)
![Zero WebRTC Blocks](https://img.shields.io/badge/Network-Firewall%20Resilient-f59e0b?style=for-the-badge)

---

## 🌟 Why Client-Side Browser Execution?

Traditional online exam proctoring software suffers from major systemic hurdles:
1. **Intrusive Software Installation:** Demanding students install kernel-level invasive agents or local Python/OpenCV environments triggers high resistance, compatibility crashes, and security vulnerabilities.
2. **University Wi-Fi Firewall Blocks:** Institutional networks regularly block or throttle high-bandwidth multi-peer WebRTC video streams.
3. **Severe Cloud Server GPU Costs:** Streaming raw HD video from 10,000 simultaneous students to central GPU servers costs thousands of dollars per hour.

**Sentinel-AI solves all three problems:**
- **Zero Local Installations:** Works directly in Chrome, Edge, Safari, or Firefox via a standard HTTPS link.
- **Hardware-Accelerated WASM:** Both `FaceLandmarker` and `ObjectDetector` run directly on the student's GPU/CPU inside the browser tab at 30–60 FPS.
- **Privacy-First:** Live video never leaves the student's device unless a verified high-severity violation triggers an encrypted visual evidence snapshot.

---

## 🎯 Core AI Detection Modules & Anti-Cheat Logic

### 1. Head Pose & 3D Gaze Attention Tracking
- Leverages MediaPipe's **478 3D facial landmark mesh** and **facial transformation rotation matrices**.
- Computes real-time **Yaw** (horizontal), **Pitch** (vertical), and **Roll** (tilt) Euler angles with Exponential Moving Average (EMA) jitter suppression.
- **"Calibrate Neutral Posture" Button:** Zeroes baseline angles to adapt to any laptop hinge angle, desk height, or posture.
- **Sustained Look-Away Alert:** If the candidate turns their head Left, Right, or looks Down continuously for more than **3.0 seconds**, an alert fires, points are deducted from the Integrity Score, and visual evidence is logged.

### 2. Candidate Absence Timer
- Continuously monitors detected face count.
- If **0 faces** are detected continuously for **> 2.0 seconds**, an "Absent from Frame" violation triggers, capturing the empty room/seat as photographic evidence and accumulating total absence duration.

### 3. Multiple People Detection
- `FaceLandmarker` is configured for up to **4 simultaneous faces** (`numFaces: 4`), paired with `ObjectDetector`'s `person` class.
- If **> 1 person/face** enters the webcam perimeter, an immediate high-severity alert is triggered.

### 4. Prohibited Device & Object Detection
- Evaluates real-time bounding boxes using EfficientDet-Lite0.
- Filters for prohibited exam items: `cell phone`, `book`, and secondary `laptop`.
- Highlights offending items in neon bounding boxes on the HUD and triggers an immediate high-severity violation.

### 5. Browser Tab / Focus Switching Detection
- Monitors browser `visibilitychange` (`document.hidden`) and `window.blur` events.
- Flags if a student switches tabs, minimizes the browser, or interacts with external applications / secondary monitors.

---

## 🛡️ 4-Stage False-Positive Suppression Filter

To prevent false alarms caused by **hanging clothes/jackets in the background** or **bare hands resting near the face/chin**, Sentinel-AI incorporates a multi-stage verification pipeline:

### Stage 1: Face-Verified Multi-Person & Motion Lock
- **Never rely on raw YOLO/ObjectDetector `person` boxes alone:** Hanging coats on doors or chairs resemble torsos. Sentinel-AI requires either:
  1. MediaPipe confirmed `> 1` distinct 3D facial meshes, OR
  2. A secondary `person` box has high confidence (`> 0.65`) AND active motion ($> 15\text{px}$ shift across frames).
- **Centroid Motion Lock:** Static background objects that shift $< 8\text{px}$ over $2.5\text{s}$ are classified as `Static Clutter (Ignored)`.

### Stage 2: Active Workspace Reach ROI
- Anchored to the candidate's primary face location.
- Bounds detection to an active working zone: from slightly above the forehead down to the desk/lap, and $2.2\times$ shoulder width.
- Detections floating far outside this zone (upper walls, ceiling, distant door hangers) are automatically suppressed.

### Stage 3: YCrCb Skin-Pixel Ratio Filter (Bare Hand Check)
- When a `cell phone` box is detected, pixel data is converted to `YCrCb` color space ($Cr \in [133, 173], Cb \in [77, 127]$).
- If the bounding box is **$> 55\%$ skin-tone pixels** and lacks a dark bezel ($Y < 50$) or illuminated screen ($Y > 185$), it is rejected as a **Bare Hand False Positive**.
- **Aspect Ratio & Area Sanity:** Rejects boxes with area $< 1.5\%$ or $> 35\%$ of the frame.
- **Class-Specific Confidence:** `cell phone` is set to $\ge 0.55$ and `book` to $\ge 0.50$.

### Stage 4: Temporal Persistence Buffer & Room Mask Calibration
- **Rolling Frame Buffer (4 of 6 Frames):** Requires an object to persist across at least 4 out of the last 6 inference frames before raising an official violation. Transient frames display an amber `ANALYZING SUSPECTED ITEM` banner without deducting integrity score.
- **1-Click Room Background Mask (2 Seconds):** Click **"Capture Room Mask (2s)"** prior to the exam to record static background clutter and mask out those coordinates for the rest of the session.

---

## 🖥️ Modern Dashboard Features

- **Left Panel (HUD Live Video Feed):**
  - Mirrored low-latency canvas overlay drawing facial landmarks, head orientation vector reticles, object bounding boxes, live FPS, and latency metrics.
  - Dynamic flashing top status banner (`NORMAL`, `WARNING`, `VIOLATION`).
  - Real-time **Yaw, Pitch, and Roll needle gauges** with directional alerts (`CENTERED`, `LOOKING LEFT`, `LOOKING RIGHT`, `LOOKING DOWN`).
  - Session controls (`Start Proctored Session`, `Calibrate Neutral Posture`, `End Exam & Export`).
  - Visual layer switches (toggle Face Mesh, Object Boxes, HUD overlays).
- **Right Panel (Telemetry & Evidence Stream):**
  - **Live Radial Integrity Score (0–100%):** Dynamic color shifts (Green: Pristine, Amber: Caution, Red: Breach) with a **5-second violation cooldown** to prevent rapid-fire log spamming.
  - **5 Real-Time KPI Cards:** Total Warnings, Look-Away Duration, Candidate Absence Time, Tab Switches, and Devices Flagged.
  - **Photographic Evidence Gallery:** Base64 JPEG watermarked snapshots captured upon high-severity incidents, with click-to-zoom modal and instant image download.
  - **1-Click Audit Export:**
    - Download RFC 4180 **CSV Audit Log**.
    - Generate official printable **PDF / Summary Certificate** formatted for print/save as PDF (`window.print()`).
  - **Slide-out Settings Drawer:** Allows examiners to customize sensitivity thresholds (Look-Away timer, absence threshold, angle limits, object confidence, sound alerts) on the fly.

---

## 📂 Project Architecture

```
ai-exam-proctor/
├── index.html          # Semantic HTML5 dashboard with HUD canvas & KPI cards
├── styles.css          # Cyber-dark enterprise theme with @media print styling
├── proctor.js          # Client-side ES module engine (MediaPipe WASM vision tasks)
├── server.py           # Optional Python FastAPI backend for telemetry & evidence storage
├── requirements.txt    # Python backend dependencies
├── Dockerfile          # Production Docker container
├── vercel.json         # 1-Click zero-config pure static hosting on Vercel
├── netlify.toml        # Netlify static hosting configuration
├── render.yaml         # Render blueprint configuration
└── README.md           # Documentation and deployment manual
```

---

## 🚀 Quickstart & Local Execution

### Option A: Pure Static Mode (Zero Python Required)
You can serve the application with any static web server (such as Node `serve`, VS Code Live Server, or Python HTTP server):

```bash
# Using Python built-in HTTP server:
python -m http.server 8000

# Or using Node.js:
npx serve .
```
Open **`http://localhost:8000`** in Google Chrome or Microsoft Edge.

### Option B: With FastAPI Telemetry & Evidence Archive Backend
Run the included Python backend to automatically persist incident snapshots to disk (`evidence_store/`) and enable server-side session reporting:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start server
python server.py
```
Open **`http://localhost:8000`**. The dashboard header will display `Cloud Sync Active (FastAPI)`.

---

## ☁️ 1-Click Cloud Deployment

### 1. Vercel (Pure Static - Instant Global CDN)
Sentinel-AI includes a preconfigured `vercel.json`.
1. Push this repository to GitHub.
2. Import the repository in [Vercel](https://vercel.com).
3. Framework Preset: **Other**.
4. Output Directory: `.` (Root).
5. Click **Deploy**.

### 2. Netlify (Static Global CDN)
Includes `netlify.toml`.
1. Drag and drop the folder into [Netlify Drop](https://app.netlify.com/drop) or connect via GitHub.
2. Publish directory: `.` (Root).
3. Deploy!

### 3. Render (Docker Web Service with FastAPI)
Includes `render.yaml` and `Dockerfile`.
1. Link your repository on [Render](https://render.com).
2. Select **Web Service** -> **Docker Runtime**.
3. Expose port `8000`.

### 4. Docker (Self-Hosted)
```bash
docker build -t sentinel-proctor .
docker run -p 8000:8000 sentinel-proctor
```

---

## ⚙️ Configuration & Threshold Tuning

Click the **Gear Icon** in the top navigation bar to open the **Proctoring Rules & Sensitivity Drawer**:

| Setting | Default | Range | Description |
|---|---|---|---|
| **Look-Away Duration Trigger** | `3.0s` | 1.0s – 6.0s | Continuous head deviation before flagging look-away |
| **Absence Timeout** | `2.0s` | 1.0s – 5.0s | Seconds of zero detected faces before absence alert |
| **Yaw Sensitivity** | `22°` | 15° – 40° | Horizontal head angle relative to calibrated neutral |
| **Pitch Down Sensitivity** | `18°` | 12° – 35° | Downward tilt angle (looking at desk, notes, phone) |
| **Object Detection Confidence** | `35%` | 20% – 80% | Confidence required for cell phone & book detection |
| **Audio Alert Tone** | `Enabled` | On / Off | Synthesized audio beep via Web Audio API |
| **Photographic Evidence Snapshots** | `Enabled` | On / Off | Automatic canvas snapshot capture on violations |
| **Anti-Cheat Tab Switching** | `Enabled` | On / Off | Flag window blur and document visibility changes |

---

## 🔒 Privacy & Compliance Architecture

- **No Continuous Video Uplink:** Webcam frames are processed in-memory via client WebGL/WASM and discarded immediately after each frame.
- **Evidence on Demand:** Only timestamped, watermarked JPEG thumbnails corresponding to confirmed rule infractions are captured.
- **FERPA & GDPR Friendly:** Eliminates storage of terabytes of biometric exam recordings on third-party servers.

---

## 📄 License
MIT License. Built for educational institutions, online learning platforms, and certification authorities.
