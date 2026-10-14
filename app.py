"""
=============================================================================
SENTINEL-AI: AI-Powered Exam Proctoring & Integrity Monitor
Streamlit Cloud Entrypoint: app.py
=============================================================================
Runs 100% within Streamlit using:
 - Two-Tier Angled-Phone Detection Pipeline directly imported from detector.py
 - Real-time browser webcam video streaming via streamlit-webrtc
 - Interactive snapshot & audit report generation
 - Full false-positive suppression (Clothes Hanger Motion Lock, Rigid Edge + Non-Skin Core)
=============================================================================
"""

import os
import time
import math
import io
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Direct import of TwoTierAngledPhoneDetector from detector.py
from detector import (
    TwoTierAngledPhoneDetector,
    CandidatePose,
    BoundingBox,
    DetectionCandidate
)

# Optional WebRTC Streamer
try:
    from streamlit_webrtc import (
        webrtc_streamer,
        VideoProcessorBase,
        RTCConfiguration,
        WebRtcMode
    )
    import av
    HAS_WEBRTC = True
except Exception as _e:
    HAS_WEBRTC = False

# =============================================================================
# Streamlit App Configuration & Dark Cyber Theme
# =============================================================================
st.set_page_config(
    page_title="Sentinel-AI: Exam Proctoring Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Dark Glassmorphism UI
st.markdown("""
<style>
  /* Global Dark Palette */
  .stApp {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }
  
  /* Metric Cards */
  div[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 12px;
    padding: 14px 18px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.35);
  }
  div[data-testid="stMetricLabel"] {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  div[data-testid="stMetricValue"] {
    color: #38bdf8;
    font-weight: 700;
    font-size: 1.8rem;
    font-family: 'JetBrains Mono', monospace;
  }
  
  /* Status Banners */
  .hud-banner {
    padding: 12px 18px;
    border-radius: 8px;
    font-weight: 700;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
  }
  .hud-normal {
    background: rgba(16, 185, 129, 0.15);
    border: 1px solid rgba(16, 185, 129, 0.4);
    color: #10b981;
  }
  .hud-warning {
    background: rgba(245, 158, 11, 0.15);
    border: 1px solid rgba(245, 158, 11, 0.4);
    color: #fbbf24;
  }
  .hud-violation {
    background: rgba(244, 63, 94, 0.18);
    border: 1px solid rgba(244, 63, 94, 0.5);
    color: #f43f5e;
    animation: pulse 1.5s infinite;
  }
  @keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(244, 63, 94, 0); }
    100% { box-shadow: 0 0 0 0 rgba(244, 63, 94, 0); }
  }

  /* Log entries */
  .log-badge-high {
    background: #f43f5e;
    color: #fff;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
  }
  .log-badge-warn {
    background: #f59e0b;
    color: #000;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
  }
  .log-badge-info {
    background: #0284c7;
    color: #fff;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 700;
  }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State Initialization
# =============================================================================
if "session_active" not in st.session_state:
    st.session_state.session_active = False
if "session_start_time" not in st.session_state:
    st.session_state.session_start_time = None
if "integrity_score" not in st.session_state:
    st.session_state.integrity_score = 100
if "total_warnings" not in st.session_state:
    st.session_state.total_warnings = 0
if "look_away_duration" not in st.session_state:
    st.session_state.look_away_duration = 0.0
if "absence_duration" not in st.session_state:
    st.session_state.absence_duration = 0.0
if "devices_flagged" not in st.session_state:
    st.session_state.devices_flagged = 0
if "incident_log" not in st.session_state:
    st.session_state.incident_log = []
if "evidence_snapshots" not in st.session_state:
    st.session_state.evidence_snapshots = []
if "baseline_pose" not in st.session_state:
    st.session_state.baseline_pose = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "calibrated": False}

# =============================================================================
# Face & Pose Detection Utilities
# =============================================================================
# =============================================================================
# Face & Pose Detection Utilities
# =============================================================================
def get_face_cascades():
    """Safely loads Haar Cascades if supported by current OpenCV build."""
    face_xml = "haarcascade_frontalface_default.xml"
    eye_xml = "haarcascade_eye.xml"
    
    fc = None
    ec = None
    if hasattr(cv2, 'CascadeClassifier'):
        try:
            if os.path.exists(face_xml):
                fc = cv2.CascadeClassifier(face_xml)
            if os.path.exists(eye_xml):
                ec = cv2.CascadeClassifier(eye_xml)
        except Exception:
            pass
    return fc, ec

face_cascade, eye_cascade = get_face_cascades()

def detect_faces_robust(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Detects faces using cascade classifier or resilient skin-morphology ellipse detector."""
    h, w = frame_bgr.shape[:2]
    
    # 1. Try Haar Cascade if available
    if face_cascade is not None:
        try:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            detected = face_cascade.detectMultiScale(gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60))
            if len(detected) > 0:
                return [(int(x), int(y), int(bw), int(bh)) for (x, y, bw, bh) in detected]
        except Exception:
            pass

    # 2. Resilient Skin + Morphological Ellipse Detector (Zero external dependency)
    ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    skin = cv2.inRange(ycrcb, np.array([0, 133, 77], np.uint8), np.array([255, 173, 127], np.uint8))
    # Candidate faces sit in the upper 80% of webcam frame
    skin[int(h * 0.80):, :] = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_clean = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, kernel)
    skin_clean = cv2.morphologyEx(skin_clean, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(skin_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    faces = []
    frame_area = float(h * w)
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if 0.03 <= (area / frame_area) <= 0.55:
            ar = float(bw) / float(bh)
            if 0.52 <= ar <= 1.45:
                faces.append((x, y, bw, bh))

    faces.sort(key=lambda f: f[2] * f[3], reverse=True)
    return faces

def estimate_head_pose_and_faces(frame_bgr: np.ndarray, baseline: dict) -> Tuple[List[Tuple[int, int, int, int]], CandidatePose, Optional[BoundingBox]]:
    """
    Detects faces, calculates real-time yaw/pitch Euler angles,
    and constructs candidate reach workspace ROI.
    """
    h, w = frame_bgr.shape[:2]
    faces = detect_faces_robust(frame_bgr)

    if not faces:
        return [], CandidatePose(0.0, 0.0, 0.0, baseline.get("calibrated", False)), None

    primary = faces[0]
    fx, fy, fw, fh = primary

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    face_gray = gray[fy:fy+fh, fx:fx+fw]
    raw_yaw = 0.0
    raw_pitch = 0.0

    # Estimate eye/pupil centroid for gaze direction
    if eye_cascade is not None:
        try:
            eyes = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=3, minSize=(15, 15))
            if len(eyes) >= 2:
                eyes_sorted = sorted(eyes, key=lambda e: e[0])
                e1, e2 = eyes_sorted[0], eyes_sorted[-1]
                eye_cx1 = fx + e1[0] + e1[2] / 2.0
                eye_cx2 = fx + e2[0] + e2[2] / 2.0
                eye_mid = (eye_cx1 + eye_cx2) / 2.0
                face_mid = fx + fw / 2.0
                offset_x = (eye_mid - face_mid) / (fw * 0.35 + 1e-5)
                raw_yaw = float(np.clip(-offset_x * 45.0, -50.0, 50.0))
        except Exception:
            pass

    rel_yaw = raw_yaw - baseline.get("yaw", 0.0)
    rel_pitch = raw_pitch - baseline.get("pitch", 0.0)
    pose = CandidatePose(yaw=rel_yaw, pitch=rel_pitch, roll=0.0, is_calibrated=baseline.get("calibrated", False))

    # Construct Active Workspace Reach ROI around candidate
    reach_w = fw * 3.8
    reach_x1 = max(0, (fx + fw/2.0) - reach_w / 2.0)
    reach_x2 = min(w, (fx + fw/2.0) + reach_w / 2.0)
    reach_y1 = max(0, fy - fh * 0.35)
    reach_y2 = min(h, fy + fh * 3.8)
    workspace_roi = BoundingBox(x1=reach_x1, y1=reach_y1, x2=reach_x2, y2=reach_y2)

    return faces, pose, workspace_roi


def extract_candidate_object_proposals(frame_bgr: np.ndarray, workspace_roi: Optional[BoundingBox]) -> List[Dict[str, Any]]:
    """
    Extracts candidate device proposals in the active workspace region
    based on high-contrast rectangular slabs, edge density, and phone aspect ratios.
    All candidates are subsequently strictly verified by TwoTierAngledPhoneDetector.
    """
    h, w = frame_bgr.shape[:2]
    proposals = []

    # Mask to workspace ROI if available
    mask = np.zeros((h, w), dtype=np.uint8)
    if workspace_roi:
        x1, y1 = int(workspace_roi.x1), int(workspace_roi.y1)
        x2, y2 = int(workspace_roi.x2), int(workspace_roi.y2)
        mask[y1:y2, x1:x2] = 255
    else:
        mask[int(h * 0.25):, :] = 255

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=mask)

    # Dilate to connect edges
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = float(h * w)

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        box_area = float(bw * bh)
        area_ratio = box_area / frame_area

        # Phone size & aspect ratio sanity check
        if 0.015 <= area_ratio <= 0.35:
            ar = float(bw) / float(bh)
            if 0.25 <= ar <= 4.0:
                # Calculate edge density & dark/screen core inside crop
                crop = frame_bgr[y:y+bh, x:x+bw]
                if crop.size == 0:
                    continue

                # Estimate base score
                edge_crop = edges[y:y+bh, x:x+bw]
                density = float(cv2.countNonZero(edge_crop)) / float(bw * bh)
                conf = float(np.clip(0.18 + density * 1.8, 0.20, 0.85))

                proposals.append({
                    "box": [float(x), float(y), float(x + bw), float(y + bh)],
                    "confidence": conf,
                    "class_id": 67,
                    "name": "cell phone"
                })

    return proposals


# =============================================================================
# Drawing & Visual HUD Overlay Helper
# =============================================================================
def draw_proctoring_hud(
    frame_bgr: np.ndarray,
    faces: List[Tuple[int, int, int, int]],
    pose: CandidatePose,
    workspace_roi: Optional[BoundingBox],
    result: Dict[str, Any],
    fps: float,
    score: int
) -> np.ndarray:
    """Renders sleek cybersecurity overlay on top of webcam frame."""
    out = frame_bgr.copy()
    h, w = out.shape[:2]

    # 1. Draw Active Reach Workspace ROI (Cyan Dashed)
    if workspace_roi:
        rx1, ry1 = int(workspace_roi.x1), int(workspace_roi.y1)
        rx2, ry2 = int(workspace_roi.x2), int(workspace_roi.y2)
        cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (248, 189, 56), 1, lineType=cv2.LINE_AA)
        cv2.putText(out, "ACTIVE WORKSPACE REACH ROI", (rx1 + 6, ry1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (248, 189, 56), 1, cv2.LINE_AA)

    # 2. Draw Face Mesh / Bounding Box
    if len(faces) == 1:
        fx, fy, fw, fh = faces[0]
        cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (56, 189, 248), 2)
        cv2.putText(out, f"CANDIDATE (1 FACE) | YAW: {pose.yaw:+.1f} deg",
                    (fx, max(18, fy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (56, 189, 248), 1, cv2.LINE_AA)
    elif len(faces) > 1:
        for (fx, fy, fw, fh) in faces:
            cv2.rectangle(out, (fx, fy), (fx + fw, fy + fh), (63, 63, 244), 2)
            cv2.putText(out, "UNAUTHORIZED PERSON", (fx, max(18, fy - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (63, 63, 244), 1, cv2.LINE_AA)

    # 3. Draw Confirmed Prohibited Devices (Neon Rose)
    for cand in result.get("confirmed_phones", []):
        b = cand.box
        bx1, by1 = int(b.x1), int(b.y1)
        bx2, by2 = int(b.x2), int(b.y2)

        tier_tag = "[TIER-2 ANGLED]" if cand.tier == 2 else "[TIER-1 DIRECT]"
        label = f"WARNING: PHONE {tier_tag} ({int(cand.confidence * 100)}%)"

        cv2.rectangle(out, (bx1, by1), (bx2, by2), (94, 63, 244), 3)
        cv2.rectangle(out, (bx1, max(0, by1 - 24)), (bx2, by1), (94, 63, 244), -1)
        cv2.putText(out, label, (bx1 + 4, max(16, by1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # 4. Draw Suppressed False Positives (Hands, Background Clutter in Slate)
    for supp in result.get("suppressed_items", []):
        b = supp.box
        sx1, sy1 = int(b.x1), int(b.y1)
        sx2, sy2 = int(b.x2), int(b.y2)
        cv2.rectangle(out, (sx1, sy1), (sx2, sy2), (148, 163, 184), 1)
        cv2.putText(out, f"[SUPPRESSED] {supp.suppression_reason}",
                    (sx1 + 4, max(12, sy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (148, 163, 184), 1, cv2.LINE_AA)

    # 5. Top Telemetry Banner
    cv2.rectangle(out, (0, 0), (w, 36), (15, 23, 42), -1)
    status_text = "CRITICAL VIOLATION" if result.get("is_phone_violation") else "STATUS: OPTIMAL"
    status_color = (94, 63, 244) if result.get("is_phone_violation") else (129, 185, 16)

    cv2.putText(out, f"SENTINEL-AI | SCORE: {score}% | {status_text} | FPS: {fps:.1f}",
                (14, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

    return out


# =============================================================================
# WebRTC Video Processor Class
# =============================================================================
if HAS_WEBRTC:
    class ProctorWebRtcProcessor(VideoProcessorBase):
        def __init__(self):
            self.detector = TwoTierAngledPhoneDetector()
            self.last_time = time.time()
            self.fps = 0.0
            self.baseline = {"yaw": 0.0, "pitch": 0.0, "calibrated": False}
            self.latest_result = {}
            self.latest_score = 100

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            now = time.time()
            dt = now - self.last_time
            if dt > 0:
                self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt)
            self.last_time = now

            img = frame.to_ndarray(format="bgr24")
            faces, pose, roi = estimate_head_pose_and_faces(img, self.baseline)
            proposals = extract_candidate_object_proposals(img, roi)

            result = self.detector.process_frame(
                frame_bgr=img,
                raw_detections=proposals,
                pose=pose,
                workspace_roi=roi,
                current_time=now
            )
            self.latest_result = result

            if result.get("is_phone_violation"):
                self.latest_score = max(0, self.latest_score - 1)

            annotated = draw_proctoring_hud(img, faces, pose, roi, result, self.fps, self.latest_score)
            return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# =============================================================================
# Sidebar Controls & Sensitivity Calibration
# =============================================================================
with st.sidebar:
    st.markdown("### 🛡️ Sentinel-AI Exam Controls")
    
    # Candidate Metadata
    cand_name = st.text_input("Candidate Name", value="Jane Doe")
    cand_id = st.text_input("Student ID", value="ST-8821")
    exam_title = st.text_input("Exam Subject", value="CS-401 Computer Vision Finals")
    
    st.markdown("---")
    st.markdown("### ⚙️ Two-Tier Sensitivity Calibration")

    base_phone_conf = st.slider("Base Phone Confidence (Tier 2 Threshold)", min_value=0.15, max_value=0.50, value=0.20, step=0.05,
                                help="Allows angled and partially occluded phone candidates into Tier-2 verification.")
    tier1_conf = st.slider("Tier-1 Direct Accept Threshold", min_value=0.40, max_value=0.80, value=0.50, step=0.05,
                           help="Detections above this confidence are accepted directly within the workspace ROI.")
    skin_rejection_ratio = st.slider("Bare Hand Skin-Tone Threshold", min_value=0.60, max_value=0.90, value=0.80, step=0.05,
                                    help="Only rejects as a bare hand if >80% skin tone AND lacks rigid straight edges or non-skin core.")
    temporal_confirm = st.slider("Temporal Confirmation Buffer", min_value=1, max_value=5, value=2, step=1,
                                 help="Requires detection in N out of 5 frames with 1.5s memory persistence.")
    
    use_motion_lock = st.checkbox("Static Background Motion Lock (<6px over 2.5s)", value=True,
                                  help="Suppresses stationary background clutter and hanging clothes.")
    use_workspace_roi = st.checkbox("Active Workspace Reach Zone Constraint", value=True,
                                    help="Ignores objects floating in the high background outside reach zone.")

    st.markdown("---")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🎯 Calibrate Neutral", use_container_width=True):
            st.session_state.baseline_pose["calibrated"] = True
            st.success("Neutral calibrated!")
    with col_btn2:
        if st.button("🔄 Reset Score", use_container_width=True):
            st.session_state.integrity_score = 100
            st.session_state.total_warnings = 0
            st.session_state.devices_flagged = 0
            st.session_state.incident_log.clear()
            st.session_state.evidence_snapshots.clear()
            st.info("Score reset to 100%")


# =============================================================================
# Main Dashboard Layout
# =============================================================================
st.title("🛡️ Sentinel-AI: Exam Proctoring & Integrity Monitor")
st.markdown(f"**Candidate:** `{cand_name}` | **ID:** `{cand_id}` | **Exam:** `{exam_title}` | **Architecture:** `Two-Tier WASM/OpenCV Engine`")

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric("Integrity Score", f"{st.session_state.integrity_score}%")
with kpi2:
    st.metric("Total Warnings", st.session_state.total_warnings)
with kpi3:
    st.metric("Look-Away", f"{st.session_state.look_away_duration:.1f}s")
with kpi4:
    st.metric("Devices Flagged", st.session_state.devices_flagged)
with kpi5:
    st.metric("Buffer Window", f"{temporal_confirm} of 5 frames")

st.markdown("---")

# Main Content: 2-Column Split
left_col, right_col = st.columns([1.25, 1.0])

# Global detector instance for manual analysis
detector = TwoTierAngledPhoneDetector(
    base_phone_conf=base_phone_conf,
    tier1_threshold=tier1_conf,
    temporal_confirm_req=temporal_confirm,
    static_motion_threshold_px=6.0,
    static_motion_duration_sec=2.5
)

with left_col:
    st.markdown("### 📹 Live Video Stream & HUD")

    tab_live, tab_snapshot = st.tabs(["🔴 Real-Time WebRTC Stream", "📸 Interactive Snapshot Analysis"])

    with tab_live:
        if HAS_WEBRTC:
            st.info("Click **START** below to stream webcam with real-time Two-Tier Angled-Phone & False-Positive HUD.")
            webrtc_ctx = webrtc_streamer(
                key="sentinel-exam-proctor",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                }),
                video_processor_factory=ProctorWebRtcProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True
            )
        else:
            st.warning("`streamlit-webrtc` is not active in this environment. Please switch to the **Snapshot Analysis** tab.")

    with tab_snapshot:
        st.markdown("Capture an instant webcam snapshot or upload an exam photo to test the **Two-Tier Angled Phone Pipeline**:")
        cam_image = st.camera_input("Take Exam Snapshot")
        uploaded_image = st.file_uploader("Or Upload Frame Image", type=["jpg", "jpeg", "png"])

        active_file = cam_image if cam_image is not None else uploaded_image

        if active_file is not None:
            file_bytes = np.asarray(bytearray(active_file.read()), dtype=np.uint8)
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if frame_bgr is not None:
                faces, pose, roi = estimate_head_pose_and_faces(frame_bgr, st.session_state.baseline_pose)
                proposals = extract_candidate_object_proposals(frame_bgr, roi)

                result = detector.process_frame(
                    frame_bgr=frame_bgr,
                    raw_detections=proposals,
                    pose=pose,
                    workspace_roi=roi
                )

                # Update score and telemetry
                if result.get("is_phone_violation"):
                    st.session_state.integrity_score = max(0, st.session_state.integrity_score - 15)
                    st.session_state.devices_flagged += 1
                    event_title = "PROHIBITED CELL PHONE DETECTED"
                    st.session_state.incident_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Severity": "HIGH",
                        "Event": event_title,
                        "Details": f"Confirmed {len(result['confirmed_phones'])} device(s) via Two-Tier Pipeline",
                        "Score": f"{st.session_state.integrity_score}%"
                    })
                    # Save snapshot
                    st.session_state.evidence_snapshots.append({
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "title": event_title,
                        "image": frame_bgr
                    })
                elif len(faces) > 1:
                    st.session_state.integrity_score = max(0, st.session_state.integrity_score - 15)
                    st.session_state.incident_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Severity": "HIGH",
                        "Event": "MULTIPLE PEOPLE DETECTED",
                        "Details": f"Observed {len(faces)} faces in exam workspace",
                        "Score": f"{st.session_state.integrity_score}%"
                    })
                elif len(faces) == 0:
                    st.session_state.integrity_score = max(0, st.session_state.integrity_score - 5)
                    st.session_state.incident_log.append({
                        "Time": datetime.now().strftime("%H:%M:%S"),
                        "Severity": "WARN",
                        "Event": "CANDIDATE ABSENT FROM FRAME",
                        "Details": "Zero faces visible in webcam frame",
                        "Score": f"{st.session_state.integrity_score}%"
                    })

                # Render annotated HUD
                annotated = draw_proctoring_hud(frame_bgr, faces, pose, roi, result, 30.0, st.session_state.integrity_score)
                rgb_annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                st.image(rgb_annotated, caption="Two-Tier False-Positive Filter Analysis", use_container_width=True)

                # Telemetry breakdown
                st.markdown(f"""
                - **Faces Detected:** `{len(faces)}`
                - **Head Pose Yaw:** `{pose.yaw:+.1f}°`
                - **Tier-2 Verified Phones:** `{result['tier2_verified_count']}`
                - **Suppressed Background Clutter / Hands:** `{len(result['suppressed_items'])}`
                - **Temporal Confirmation Status:** `{result['temporal_confirm_status']}`
                """)


with right_col:
    st.markdown("### 📋 Audit Telemetry & Incident Log")

    if st.session_state.incident_log:
        df_log = pd.DataFrame(st.session_state.incident_log)
        st.dataframe(df_log, use_container_width=True)

        # Download CSV Audit Log
        csv_buffer = io.StringIO()
        df_log.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download RFC 4180 Audit Log (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"Proctor_Audit_{cand_id}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Zero incidents recorded. Candidate maintaining optimal exam integrity.")

    st.markdown("### 📸 Visual Evidence Gallery")
    if st.session_state.evidence_snapshots:
        cols = st.columns(min(3, len(st.session_state.evidence_snapshots)))
        for idx, snap in enumerate(st.session_state.evidence_snapshots[-3:]):
            with cols[idx % len(cols)]:
                rgb_snap = cv2.cvtColor(snap["image"], cv2.COLOR_BGR2RGB)
                st.image(rgb_snap, caption=f"{snap['title']} ({snap['time']})", use_container_width=True)
    else:
        st.caption("Visual snapshot evidence will appear here when an infraction occurs.")

    st.markdown("---")
    st.markdown("""
    **Pipeline Architecture:**
    - `detector.py`: Two-Tier Angled Phone Pipeline (Canny Edge + Non-Skin Core + Motion Lock)
    - `app.py`: Streamlit Community Cloud UI & WebRTC Video Engine
    - `server.py`: Standalone FastAPI Backend & REST API
    """)
