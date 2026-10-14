"""
=============================================================================
SENTINEL-AI: AI-Powered Exam Proctoring & Integrity Monitor
Lightweight Python FastAPI Backend (Telemetry Aggregator & Evidence Archive)
=============================================================================
"""

import os
import re
import csv
import base64
import io
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
from pydantic import BaseModel, Field

import cv2
import numpy as np

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent
EVIDENCE_DIR = BASE_DIR / "evidence_store"
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize Two-Tier Angled-Phone Vision Engine
try:
    from detector import TwoTierAngledPhoneDetector, CandidatePose, BoundingBox
    server_detector = TwoTierAngledPhoneDetector()
except Exception as _e:
    server_detector = None
    print(f"Warning: Could not initialize server-side TwoTierAngledPhoneDetector: {_e}")

app = FastAPI(
    title="Sentinel-AI Proctoring Backend",
    description="Telemetry Aggregation and Photographic Evidence Storage for AI Exam Proctoring",
    version="2.0.0"
)

# Enable CORS for local testing and cross-origin cloud deployments
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Database for Active Sessions
sessions_db: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# Pydantic Schemas
# =============================================================================
class CandidateInfo(BaseModel):
    name: str = "Jane Doe"
    id: str = "ST-8821"
    examTitle: str = "CS-401 Computer Vision & AI Finals"

class StartSessionRequest(BaseModel):
    sessionId: str
    candidate: CandidateInfo
    startTime: Optional[str] = None

class EvidenceThumb(BaseModel):
    id: Optional[str] = None
    timestamp: Optional[str] = None
    date: Optional[str] = None
    elapsed: Optional[str] = None
    title: Optional[str] = None
    details: Optional[str] = None
    severity: Optional[str] = None
    score: Optional[int] = None
    imageData: Optional[str] = None

class SessionEventRequest(BaseModel):
    sessionId: str
    candidate: CandidateInfo
    type: str
    severity: str
    title: str
    details: str
    integrityScore: int
    evidenceThumb: Optional[EvidenceThumb] = None

class VerifyFrameRequest(BaseModel):
    image: str
    detections: List[Dict[str, Any]] = []
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    workspace_roi: Optional[List[float]] = None

# =============================================================================
# API Endpoints
# =============================================================================
@app.get("/api/health")
async def health_check():
    """Health check endpoint to verify backend availability."""
    return {
        "status": "online",
        "service": "Sentinel-AI Proctoring Engine",
        "version": "2.0.0",
        "mode": "Client-Side WASM Vision + Python Telemetry Sync",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/verify/angled-phone")
async def verify_angled_phone_endpoint(payload: VerifyFrameRequest):
    """
    Server-side Two-Tier Angled-Phone Verification using OpenCV Canny, HoughLinesP,
    YCrCb Skin Masking, and Non-Skin Core analysis.
    """
    if server_detector is None:
        raise HTTPException(status_code=503, detail="Server-side vision detector not loaded")

    try:
        img_str = payload.image
        if "," in img_str:
            img_str = img_str.split(",")[1]
        img_bytes = base64.b64decode(img_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame_bgr is None:
            raise HTTPException(status_code=400, detail="Failed to decode image frame")

        pose = CandidatePose(yaw=payload.yaw, pitch=payload.pitch, roll=payload.roll)
        roi = None
        if payload.workspace_roi and len(payload.workspace_roi) == 4:
            roi = BoundingBox(*payload.workspace_roi)

        result = server_detector.process_frame(
            frame_bgr=frame_bgr,
            raw_detections=payload.detections,
            pose=pose,
            workspace_roi=roi
        )

        confirmed = [
            {
                "box": [c.box.x1, c.box.y1, c.box.x2, c.box.y2],
                "confidence": c.confidence,
                "tier": c.tier,
                "has_rigid_edge": c.has_rigid_edge,
                "has_phone_core": c.has_phone_core,
                "skin_ratio": c.skin_ratio,
                "core_ratio": c.core_ratio,
                "context_boosted": c.context_boosted
            }
            for c in result["confirmed_phones"]
        ]
        suppressed = [
            {
                "box": [s.box.x1, s.box.y1, s.box.x2, s.box.y2],
                "reason": s.suppression_reason,
                "confidence": s.confidence
            }
            for s in result["suppressed_items"]
        ]

        return {
            "status": "success",
            "is_phone_violation": result["is_phone_violation"],
            "temporal_confirm_status": result["temporal_confirm_status"],
            "tier2_verified_count": result["tier2_verified_count"],
            "confirmed_phones": confirmed,
            "suppressed_items": suppressed,
            "memory_active": result["memory_active"]
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))

@app.post("/api/session/start")
async def start_session(payload: StartSessionRequest):
    """Initializes and records a new proctored exam session."""
    session_id = payload.sessionId
    session_dir = EVIDENCE_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    sessions_db[session_id] = {
        "sessionId": session_id,
        "candidate": payload.candidate.dict(),
        "startTime": payload.startTime or datetime.utcnow().isoformat(),
        "endTime": None,
        "finalScore": 100,
        "events": [],
        "evidenceCount": 0
    }

    return {
        "status": "success",
        "message": f"Session {session_id} registered successfully",
        "session": sessions_db[session_id]
    }

@app.post("/api/session/event")
async def record_session_event(payload: SessionEventRequest):
    """Receives and stores an audit infraction or telemetry event with optional visual snapshot."""
    session_id = payload.sessionId

    if session_id not in sessions_db:
        sessions_db[session_id] = {
            "sessionId": session_id,
            "candidate": payload.candidate.dict(),
            "startTime": datetime.utcnow().isoformat(),
            "endTime": None,
            "finalScore": payload.integrityScore,
            "events": [],
            "evidenceCount": 0
        }

    session = sessions_db[session_id]
    session["finalScore"] = payload.integrityScore

    saved_image_path = None

    # Handle Photographic Visual Evidence Storage
    if payload.evidenceThumb and payload.evidenceThumb.imageData:
        try:
            image_b64 = payload.evidenceThumb.imageData
            if "," in image_b64:
                image_b64 = image_b64.split(",")[1]

            image_bytes = base64.b64decode(image_b64)
            timestamp_safe = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"ev_{timestamp_safe}_{payload.type.lower()}.jpg"
            
            session_dir = EVIDENCE_DIR / session_id
            session_dir.mkdir(parents=True, exist_ok=True)
            file_path = session_dir / filename

            with open(file_path, "wb") as f:
                f.write(image_bytes)

            saved_image_path = f"/evidence/{session_id}/{filename}"
            session["evidenceCount"] += 1
        except Exception as err:
            print(f"Error saving evidence image: {err}")

    event_record = {
        "eventIndex": len(session["events"]) + 1,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "type": payload.type,
        "severity": payload.severity,
        "title": payload.title,
        "details": payload.details,
        "integrityScore": payload.integrityScore,
        "evidencePath": saved_image_path
    }

    session["events"].append(event_record)

    return {
        "status": "recorded",
        "eventIndex": event_record["eventIndex"],
        "evidenceSaved": bool(saved_image_path)
    }

@app.get("/api/session/{session_id}/report")
async def get_session_report(session_id: str):
    """Retrieves full proctoring telemetry audit report for a session."""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    return sessions_db[session_id]

@app.get("/api/session/{session_id}/export.csv")
async def export_session_csv(session_id: str):
    """Streams a RFC 4180 compliant CSV audit log for download."""
    if session_id not in sessions_db:
        raise HTTPException(status_code=404, detail="Session not found")

    session = sessions_db[session_id]
    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Event Index",
        "Session ID",
        "Candidate Name",
        "Candidate ID",
        "Exam Title",
        "Timestamp",
        "Severity",
        "Violation Title",
        "Details",
        "Integrity Score",
        "Evidence Saved"
    ])

    for ev in session["events"]:
        writer.writerow([
            ev["eventIndex"],
            session_id,
            session["candidate"]["name"],
            session["candidate"]["id"],
            session["candidate"]["examTitle"],
            ev["timestamp"],
            ev["severity"],
            ev["title"],
            ev["details"],
            ev["integrityScore"],
            "YES" if ev["evidencePath"] else "NO"
        ])

    csv_data = output.getvalue()
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=Proctor_Audit_{session_id}.csv"}
    )

@app.get("/api/sessions")
async def list_all_sessions():
    """Lists summary of all registered sessions."""
    summaries = []
    for s_id, data in sessions_db.items():
        summaries.append({
            "sessionId": s_id,
            "candidate": data["candidate"],
            "startTime": data["startTime"],
            "finalScore": data["finalScore"],
            "totalIncidents": len(data["events"]),
            "evidenceCount": data["evidenceCount"]
        })
    return {"sessions": summaries}

# Serve captured evidence images
if EVIDENCE_DIR.exists():
    app.mount("/evidence", StaticFiles(directory=str(EVIDENCE_DIR)), name="evidence")

# Serve client-side static web application files directly
app.mount("/", StaticFiles(directory=str(BASE_DIR), html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print("🛡️  SENTINEL-AI PROCTORING SERVER STARTING")
    print("📍 Local UI Dashboard: http://localhost:8000")
    print("⚡ WebAssembly Models: Loaded dynamically client-side in browser")
    print("📦 Storage Directory: ", EVIDENCE_DIR)
    print("="*70 + "\n")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
