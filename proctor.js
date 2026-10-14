/**
 * ============================================================================
 * SENTINEL-AI: AI-Powered Exam Proctoring & Integrity Monitor
 * 100% Client-Side WebAssembly / WebGL Computer Vision Engine
 * Powered by Google MediaPipe Tasks Vision (FaceLandmarker & ObjectDetector)
 * Upgraded with 4-Stage False-Positive Suppression Pipeline:
 *  1. Face-Verified Multi-Person & Motion Lock (Clothes on Hangers Ignore)
 *  2. Active Workspace ROI (Candidate Reach Zone Constraint)
 *  3. YCrCb Skin-Pixel Ratio Filter & Aspect-Ratio Sanity (Hand vs Phone Check)
 *  4. Temporal Persistence Buffer (4 of 6 Frames) & Room Background Masking
 * ============================================================================
 */

import {
  FilesetResolver,
  FaceLandmarker,
  ObjectDetector,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

// ============================================================================
// State Management
// ============================================================================
const state = {
  // Vision Models
  vision: null,
  faceLandmarker: null,
  objectDetector: null,
  drawingUtils: null,
  modelsReady: false,

  // Stream & Canvas
  video: null,
  canvas: null,
  ctx: null,
  stream: null,
  animationFrameId: null,
  lastVideoTime: -1,
  canvasWidth: 640,
  canvasHeight: 360,

  // Session Control
  sessionActive: false,
  sessionStartTime: null,
  sessionElapsedTime: 0,
  sessionTimerInterval: null,
  sessionId: "SES-" + Math.random().toString(36).substring(2, 8).toUpperCase(),

  // Candidate Details
  candidate: {
    name: "Jane Doe",
    id: "ST-8821",
    examTitle: "CS-401 Computer Vision & AI Finals"
  },

  // Configuration Thresholds & Two-Tier Angled-Phone Pipeline
  config: {
    lookAwayTime: 3.0,          // Sustained look-away alert trigger (seconds)
    absenceTime: 2.0,           // Absent from frame alert trigger (seconds)
    yawThreshold: 22,           // Degrees relative to calibrated baseline
    pitchThreshold: 18,         // Downward tilt degrees relative to calibrated baseline
    objConf: 0.20,              // Base object detector confidence threshold (imgsz 640 equivalent)
    phoneConf: 0.20,            // Base threshold for cell phone (enables Tier-2 angled verification)
    tier1Conf: 0.50,            // Tier 1 direct acceptance threshold
    bookConf: 0.45,             // Class-specific threshold for books
    useSkinFilter: true,        // Rigid Edge + Non-Skin Core Filter
    skinRatioThreshold: 0.80,   // Only reject pure hands (>80% skin and no edge/core)
    useMotionLock: true,        // Static background motion lock (< 6px over 2.5s)
    useWorkspaceROI: true,      // Active workspace reach zone constraint
    temporalConfirmFrames: 2,   // 2 of last 5 frames required for rapid angled glance confirmation
    audioAlerts: true,          // Web Audio chime
    snapshotsEnabled: true,     // Capture photographic evidence
    antiCheatTab: true,         // Browser tab switch tracking
    showMesh: true,             // Render face mesh
    showBoxes: true,            // Render object bounding boxes
    showROI: true,              // Render candidate workspace ROI
    showSuppressed: true,       // Render suppressed false positive boxes
    showHUD: true               // Render telemetry HUD
  },

  // Neutral Calibration Baseline Offsets (Zeroed for laptop tilt)
  calibration: {
    calibrated: false,
    baselineYaw: 0,
    baselinePitch: 0,
    baselineRoll: 0
  },

  // Two-Tier Angled-Phone Suppression State
  suppression: {
    trackedObjects: new Map(), // Centroid tracking for static background motion lock
    temporalHistory: [],       // Rolling frame buffer (last 5 inference frames)
    lastConfirmedMemory: null, // 1.5s bounding-box memory persistence
    roomMaskZones: [],         // Calibrated static background fixture zones
    isCapturingRoomMask: false,
    roomMaskStartTime: null,
    roomMaskDuration: 2000,    // 2.0 seconds
    roomMaskSampledZones: [],
    activeWorkspaceROI: null,  // Active candidate reach box
    suppressedItems: []        // Suppressed clutter/hand items for HUD rendering
  },

  // Real-time Smoothed Angles (EMA Filter)
  pose: {
    yaw: 0,
    pitch: 0,
    roll: 0,
    relYaw: 0,
    relPitch: 0,
    relRoll: 0,
    direction: "CENTERED"
  },

  // Integrity & Violation Tracking
  integrityScore: 100,
  totalWarnings: 0,
  totalLookAwayDuration: 0,
  totalAbsenceDuration: 0,
  tabSwitchesCount: 0,
  devicesFlaggedCount: 0,

  // Real-time Event Timers
  lookAwayStartTime: null,
  absenceStartTime: null,
  sustainedLookAwayTriggered: false,
  sustainedAbsenceTriggered: false,

  // Violation Cooldowns (5 seconds cooldown per violation type)
  cooldowns: {
    LOOK_AWAY: 0,
    ABSENT: 0,
    MULTIPLE_PEOPLE: 0,
    PROHIBITED_OBJECT: 0,
    TAB_SWITCH: 0
  },

  // Telemetry & Evidence Archives
  incidentLog: [],
  evidenceSnapshots: [],
  selectedEvidence: null,

  // Performance Telemetry
  fps: 0,
  fpsCounter: 0,
  fpsLastTime: performance.now(),
  lastInferenceLatency: 0,

  // Cloud Backend Sync Status
  backendConnected: false,
  backendUrl: window.location.origin
};

// ============================================================================
// Web Audio API Alert Synthesizer (Zero External Dependencies)
// ============================================================================
class AudioSynthesizer {
  constructor() {
    this.ctx = null;
  }

  init() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      if (AudioCtx) {
        this.ctx = new AudioCtx();
      }
    }
    if (this.ctx && this.ctx.state === "suspended") {
      this.ctx.resume();
    }
  }

  playTone(freq, type = "sine", duration = 0.2, gain = 0.15) {
    if (!state.config.audioAlerts) return;
    try {
      this.init();
      if (!this.ctx) return;
      const osc = this.ctx.createOscillator();
      const gainNode = this.ctx.createGain();
      osc.type = type;
      osc.frequency.setValueAtTime(freq, this.ctx.currentTime);
      gainNode.gain.setValueAtTime(gain, this.ctx.currentTime);
      gainNode.gain.exponentialRampToValueAtTime(0.001, this.ctx.currentTime + duration);
      osc.connect(gainNode);
      gainNode.connect(this.ctx.destination);
      osc.start();
      osc.stop(this.ctx.currentTime + duration);
    } catch (e) {
      console.warn("Audio chime prevented:", e);
    }
  }

  playWarningChime() {
    this.playTone(650, "triangle", 0.25, 0.2);
    setTimeout(() => this.playTone(850, "triangle", 0.35, 0.25), 180);
  }

  playViolationAlarm() {
    this.playTone(900, "sawtooth", 0.18, 0.3);
    setTimeout(() => this.playTone(600, "sawtooth", 0.25, 0.3), 150);
  }

  playSuccessChime() {
    this.playTone(523.25, "sine", 0.15, 0.2); // C5
    setTimeout(() => this.playTone(659.25, "sine", 0.15, 0.2), 120); // E5
    setTimeout(() => this.playTone(783.99, "sine", 0.3, 0.25), 240); // G5
  }
}

const audioSynth = new AudioSynthesizer();

// ============================================================================
// DOM Elements Cache
// ============================================================================
const dom = {
  video: document.getElementById("webcamVideo"),
  canvas: document.getElementById("outputCanvas"),
  loaderOverlay: document.getElementById("loaderOverlay"),
  loaderProgressFill: document.getElementById("loaderProgressFill"),
  loaderTitle: document.getElementById("loaderTitle"),
  loaderMessage: document.getElementById("loaderMessage"),
  
  hudStatusBanner: document.getElementById("hudStatusBanner"),
  hudStatusText: document.getElementById("hudStatusText"),
  hudStatusSubtext: document.getElementById("hudStatusSubtext"),
  hudFps: document.getElementById("hudFps"),
  hudLatency: document.getElementById("hudLatency"),
  hudFaceCount: document.getElementById("hudFaceCount"),
  hudObjectCount: document.getElementById("hudObjectCount"),
  hudDirectionBadge: document.getElementById("hudDirectionBadge"),
  
  lookawayBadge: document.getElementById("lookawayBadge"),
  lookawayCountdown: document.getElementById("lookawayCountdown"),
  lookawayProgressFill: document.getElementById("lookawayProgressFill"),

  btnStartSession: document.getElementById("btnStartSession"),
  btnStartSessionText: document.getElementById("btnStartSessionText"),
  btnCalibrate: document.getElementById("btnCalibrate"),
  btnCaptureBackgroundMask: document.getElementById("btnCaptureBackgroundMask"),
  btnCaptureMaskText: document.getElementById("btnCaptureMaskText"),
  btnEndSession: document.getElementById("btnEndSession"),
  
  chkShowMesh: document.getElementById("chkShowMesh"),
  chkShowBoxes: document.getElementById("chkShowBoxes"),
  chkShowROI: document.getElementById("chkShowROI"),
  chkShowSuppressed: document.getElementById("chkShowSuppressed"),
  chkShowHUD: document.getElementById("chkShowHUD"),
  
  roomMaskActivePill: document.getElementById("roomMaskActivePill"),
  roomMaskCount: document.getElementById("roomMaskCount"),
  roomMaskOverlay: document.getElementById("roomMaskOverlay"),
  maskScanProgressFill: document.getElementById("maskScanProgressFill"),
  maskScanTitle: document.getElementById("maskScanTitle"),
  maskScanSub: document.getElementById("maskScanSub"),

  yawVal: document.getElementById("yawVal"),
  yawNeedle: document.getElementById("yawNeedle"),
  pitchVal: document.getElementById("pitchVal"),
  pitchNeedle: document.getElementById("pitchNeedle"),
  rollVal: document.getElementById("rollVal"),
  rollNeedle: document.getElementById("rollNeedle"),
  postureStatusPill: document.getElementById("postureStatusPill"),

  integrityScoreVal: document.getElementById("integrityScoreVal"),
  scoreCircleProgress: document.getElementById("scoreCircleProgress"),
  integrityVerdictBadge: document.getElementById("integrityVerdictBadge"),
  integrityDescription: document.getElementById("integrityDescription"),
  cooldownIndicator: document.getElementById("cooldownIndicator"),
  cooldownText: document.getElementById("cooldownText"),

  kpiTotalWarnings: document.getElementById("kpiTotalWarnings"),
  kpiLookAwayDuration: document.getElementById("kpiLookAwayDuration"),
  kpiAbsenceDuration: document.getElementById("kpiAbsenceDuration"),
  kpiTabSwitches: document.getElementById("kpiTabSwitches"),
  kpiDevicesCount: document.getElementById("kpiDevicesCount"),

  logStreamContainer: document.getElementById("logStreamContainer"),
  emptyLogState: document.getElementById("emptyLogState"),
  logItemsList: document.getElementById("logItemsList"),
  
  countFilterAll: document.getElementById("countFilterAll"),
  countFilterHigh: document.getElementById("countFilterHigh"),
  countFilterWarn: document.getElementById("countFilterWarn"),
  countFilterEvidence: document.getElementById("countFilterEvidence"),

  btnExportCSV: document.getElementById("btnExportCSV"),
  btnExportPDF: document.getElementById("btnExportPDF"),
  
  // Settings Drawer
  settingsDrawer: document.getElementById("settingsDrawer"),
  settingsBackdrop: document.getElementById("settingsBackdrop"),
  btnSettingsOpen: document.getElementById("btnSettingsOpen"),
  btnSettingsClose: document.getElementById("btnSettingsClose"),
  btnSettingsSave: document.getElementById("btnSettingsSave"),
  btnSettingsReset: document.getElementById("btnSettingsReset"),
  
  cfgCandidateName: document.getElementById("cfgCandidateName"),
  cfgCandidateId: document.getElementById("cfgCandidateId"),
  cfgExamTitle: document.getElementById("cfgExamTitle"),
  cfgLookAwayTime: document.getElementById("cfgLookAwayTime"),
  cfgLookAwayTimeVal: document.getElementById("cfgLookAwayTimeVal"),
  cfgAbsenceTime: document.getElementById("cfgAbsenceTime"),
  cfgAbsenceTimeVal: document.getElementById("cfgAbsenceTimeVal"),
  cfgYawThreshold: document.getElementById("cfgYawThreshold"),
  cfgYawThresholdVal: document.getElementById("cfgYawThresholdVal"),
  cfgPitchThreshold: document.getElementById("cfgPitchThreshold"),
  cfgPitchThresholdVal: document.getElementById("cfgPitchThresholdVal"),
  cfgPhoneConf: document.getElementById("cfgPhoneConf"),
  cfgPhoneConfVal: document.getElementById("cfgPhoneConfVal"),
  cfgBookConf: document.getElementById("cfgBookConf"),
  cfgBookConfVal: document.getElementById("cfgBookConfVal"),
  cfgSkinFilter: document.getElementById("cfgSkinFilter"),
  cfgSkinRatio: document.getElementById("cfgSkinRatio"),
  cfgSkinRatioVal: document.getElementById("cfgSkinRatioVal"),
  cfgMotionLock: document.getElementById("cfgMotionLock"),
  cfgWorkspaceROI: document.getElementById("cfgWorkspaceROI"),
  cfgTemporalConfirm: document.getElementById("cfgTemporalConfirm"),
  cfgTemporalConfirmVal: document.getElementById("cfgTemporalConfirmVal"),
  cfgRoomMaskSummary: document.getElementById("cfgRoomMaskSummary"),
  btnClearRoomMask: document.getElementById("btnClearRoomMask"),
  cfgAudioAlerts: document.getElementById("cfgAudioAlerts"),
  cfgSnapshotsEnabled: document.getElementById("cfgSnapshotsEnabled"),
  cfgAntiCheatTab: document.getElementById("cfgAntiCheatTab"),

  // Audio Toggle
  btnAudioToggle: document.getElementById("btnAudioToggle"),
  audioIconOn: document.getElementById("audioIconOn"),
  audioIconOff: document.getElementById("audioIconOff"),

  // Header Elements
  headerCandidateName: document.getElementById("headerCandidateName"),
  sessionTimer: document.getElementById("sessionTimer"),
  recDot: document.getElementById("recDot"),
  backendStatusPill: document.getElementById("backendStatusPill"),
  backendDot: document.getElementById("backendDot"),
  backendStatusText: document.getElementById("backendStatusText"),

  // Evidence Modal
  evidenceModalBackdrop: document.getElementById("evidenceModalBackdrop"),
  modalEvidenceTitle: document.getElementById("modalEvidenceTitle"),
  modalEvidenceImage: document.getElementById("modalEvidenceImage"),
  modalEvidenceTime: document.getElementById("modalEvidenceTime"),
  modalEvidenceElapsed: document.getElementById("modalEvidenceElapsed"),
  modalEvidenceScore: document.getElementById("modalEvidenceScore"),
  modalEvidenceDetails: document.getElementById("modalEvidenceDetails"),
  btnModalClose: document.getElementById("btnModalClose"),
  btnModalDone: document.getElementById("btnModalDone"),
  btnDownloadEvidenceImage: document.getElementById("btnDownloadEvidenceImage"),

  printReportContainer: document.getElementById("printReportContainer")
};

// ============================================================================
// Initialization & WebAssembly Model Loading
// ============================================================================
async function initApplication() {
  state.video = dom.video;
  state.canvas = dom.canvas;
  state.ctx = dom.canvas.getContext("2d", { willReadFrequently: true });

  bindEventListeners();
  checkCloudBackend();

  try {
    updateLoader(20, "Resolving WebAssembly Vision Runtime...", "Loading @mediapipe/tasks-vision from CDN");
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
    );
    state.vision = vision;

    updateLoader(55, "Loading FaceLandmarker & Pose Matrix Engine...", "Downloading multi-face 3D mesh model (float16)");
    state.faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU"
      },
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
      runningMode: "VIDEO",
      numFaces: 4
    });

    updateLoader(85, "Loading ObjectDetector (Prohibited Items)...", "Downloading EfficientDet-Lite0 model");
    state.objectDetector = await ObjectDetector.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite",
        delegate: "GPU"
      },
      scoreThreshold: 0.18, // Lower raw threshold so Tier-2 angled candidates (0.20+) are captured
      runningMode: "VIDEO"
    });

    state.drawingUtils = new DrawingUtils(state.ctx);
    state.modelsReady = true;

    updateLoader(100, "Starting Camera Stream...", "Requesting camera authorization");
    await startCameraStream();

    // Hide loader overlay with smooth fade
    dom.loaderOverlay.classList.add("fade-out");
    setTimeout(() => {
      dom.loaderOverlay.classList.add("hidden");
    }, 450);

    setHUDStatus("normal", "READY - SYSTEM INITIALIZED", "Click 'Start Proctored Session' or 'Capture Room Mask' to begin");
    logEvent("INFO", "SYSTEM_INIT", "Proctoring Vision Engine & 4-Stage Suppression Pipeline initialized", false);

  } catch (err) {
    console.error("Initialization error:", err);
    dom.loaderTitle.textContent = "Initialization Notice";
    dom.loaderMessage.innerHTML = `
      <span style="color:#f43f5e; font-weight:700;">Camera or Model Error:</span> ${err.message || err}<br><br>
      Please ensure you have granted camera permissions in your browser and are connected to the internet.
    `;
    setHUDStatus("violation", "CAMERA OR MODEL ERROR", "Check camera access permissions and refresh.");
  }
}

function updateLoader(percentage, title, message) {
  if (dom.loaderProgressFill) dom.loaderProgressFill.style.width = `${percentage}%`;
  if (dom.loaderTitle) dom.loaderTitle.textContent = title;
  if (dom.loaderMessage) dom.loaderMessage.textContent = message;
}

// ============================================================================
// Camera Stream Access
// ============================================================================
async function startCameraStream() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error("Browser does not support navigator.mediaDevices.getUserMedia");
  }

  const stream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 1280 },
      height: { ideal: 720 },
      facingMode: "user"
    },
    audio: false
  });

  state.stream = stream;
  dom.video.srcObject = stream;

  return new Promise((resolve) => {
    dom.video.onloadedmetadata = () => {
      dom.video.play();
      state.canvasWidth = dom.video.videoWidth || 1280;
      state.canvasHeight = dom.video.videoHeight || 720;
      dom.canvas.width = state.canvasWidth;
      dom.canvas.height = state.canvasHeight;
      requestAnimationFrame(renderLoop);
      resolve();
    };
  });
}

// ============================================================================
// Core Detection & Render Loop (Hardware Accelerated)
// ============================================================================
let objectDetectInterval = 0;
let cachedObjectDetections = [];

function renderLoop() {
  state.animationFrameId = requestAnimationFrame(renderLoop);

  if (!state.modelsReady || !dom.video || dom.video.readyState < 2) {
    return;
  }

  const currentTime = performance.now();

  // Calculate FPS
  state.fpsCounter++;
  if (currentTime - state.fpsLastTime >= 1000) {
    state.fps = state.fpsCounter;
    state.fpsCounter = 0;
    state.fpsLastTime = currentTime;
    dom.hudFps.textContent = state.fps;
  }

  const inferenceStart = performance.now();

  // Run FaceLandmarker inference
  let faceResult = null;
  try {
    faceResult = state.faceLandmarker.detectForVideo(dom.video, currentTime);
  } catch (err) {
    // Non-monotonic timestamp safeguard
  }

  // Run ObjectDetector inference every 3 frames for optimal performance
  objectDetectInterval++;
  if (objectDetectInterval % 3 === 0) {
    try {
      const objResult = state.objectDetector.detectForVideo(dom.video, currentTime);
      if (objResult && objResult.detections) {
        cachedObjectDetections = objResult.detections;
      }
    } catch (err) {
      // Ignored non-monotonic timestamp
    }
  }

  state.lastInferenceLatency = Math.round(performance.now() - inferenceStart);
  dom.hudLatency.textContent = `${state.lastInferenceLatency}ms`;

  // Draw Camera Frame to Canvas (mirrored for natural reflection)
  const ctx = state.ctx;
  ctx.save();
  ctx.clearRect(0, 0, state.canvasWidth, state.canvasHeight);
  ctx.translate(state.canvasWidth, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(dom.video, 0, 0, state.canvasWidth, state.canvasHeight);
  ctx.restore();

  const numFaces = (faceResult && faceResult.faceLandmarks) ? faceResult.faceLandmarks.length : 0;
  dom.hudFaceCount.textContent = numFaces;

  // Process Head Pose for Primary Face (index 0)
  if (numFaces > 0) {
    const landmarks = faceResult.faceLandmarks[0];
    const matrix = (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes[0]) 
      ? faceResult.facialTransformationMatrixes[0] 
      : null;

    calculateHeadPose(landmarks, matrix);

    // Compute Active Workspace ROI
    state.suppression.activeWorkspaceROI = getCandidateWorkspaceROI(landmarks);

    // Draw Face Mesh if toggled
    if (state.config.showMesh) {
      drawFaceLandmarksCustom(landmarks);
    }
  } else {
    state.suppression.activeWorkspaceROI = null;
  }

  // ==========================================================================
  // STAGE 1 - 4: False-Positive Suppression & Detection Pipeline
  // ==========================================================================
  const {
    activeProhibitedObjects,
    pendingConfirmationObjects,
    suppressedObjects,
    isMultiPersonConfirmed,
    multiPersonPendingCount
  } = processDetectionsWithSuppression(cachedObjectDetections, faceResult, currentTime);

  state.suppression.suppressedItems = suppressedObjects;
  dom.hudObjectCount.textContent = activeProhibitedObjects.length;

  // Handle Room Mask Calibration Sampling Phase
  if (state.suppression.isCapturingRoomMask) {
    handleRoomMaskSampling(cachedObjectDetections, currentTime);
  }

  // Draw Active Workspace ROI boundary if toggled
  if (state.config.showROI && state.suppression.activeWorkspaceROI) {
    drawWorkspaceROI(state.suppression.activeWorkspaceROI);
  }

  // Draw Suppressed Clutter & Hand items if toggled
  if (state.config.showSuppressed && suppressedObjects.length > 0) {
    drawSuppressedObjects(suppressedObjects);
  }

  // Draw Object Detection Bounding Boxes
  if (state.config.showBoxes && (activeProhibitedObjects.length > 0 || pendingConfirmationObjects.length > 0)) {
    drawProhibitedObjects(activeProhibitedObjects, pendingConfirmationObjects);
  }

  // Update Telemetry Needles & Direction Badge
  updateAngleTelemetryUI();

  // If proctored session is running, evaluate anti-cheat rules
  if (state.sessionActive) {
    evaluateIntegrityRules(
      numFaces,
      isMultiPersonConfirmed,
      multiPersonPendingCount,
      activeProhibitedObjects,
      pendingConfirmationObjects
    );
  }

  // Draw HUD Graphic Reticles & Compass if toggled
  if (state.config.showHUD && numFaces > 0) {
    drawHUDTargetReticle(faceResult.faceLandmarks[0]);
  }
}

// ============================================================================
// STAGE 1 - 4: Comprehensive False-Positive Suppression Pipeline
// ============================================================================
function processDetectionsWithSuppression(rawDetections, faceResult, now) {
  const activeProhibitedObjects = [];
  const pendingConfirmationObjects = [];
  const suppressedObjects = [];

  const numFaces = (faceResult && faceResult.faceLandmarks) ? faceResult.faceLandmarks.length : 0;
  const roi = state.suppression.activeWorkspaceROI;

  let rawPersonDetections = [];
  let rawPhoneCandidates = [];
  let rawBookCandidates = [];

  // Parse raw detections into mirrored canvas coordinates
  if (rawDetections && rawDetections.length > 0) {
    for (const det of rawDetections) {
      const category = det.categories[0];
      if (!category) continue;
      const catName = category.categoryName.toLowerCase();
      const score = category.score;
      const b = det.boundingBox;

      // Transform bounding box to mirrored canvas coordinates
      const boxX = (1.0 - (b.originX + b.width) / state.canvasWidth) * state.canvasWidth;
      const boxY = b.originY;
      const boxW = b.width;
      const boxH = b.height;

      const item = {
        name: category.categoryName,
        catName,
        score: Math.round(score * 100),
        rawScore: score,
        box: { x: boxX, y: boxY, w: boxW, h: boxH, originBox: b },
        cx: boxX + boxW / 2,
        cy: boxY + boxH / 2
      };

      if (catName === "person") {
        rawPersonDetections.push(item);
      } else if (catName === "cell phone") {
        rawPhoneCandidates.push(item);
      } else if (catName === "book") {
        rawBookCandidates.push(item);
      } else if (catName === "laptop" && score >= 0.50) {
        // Evaluate secondary laptop
        activeProhibitedObjects.push(item);
      }
    }
  }

  // Cheating Posture Check: glance downward (relative pitch > 8° or < -8°) or head turned (|relative yaw| > 10°)
  const isCheatingPose = (
    Math.abs(state.pose.relPitch) > 8 ||
    Math.abs(state.pose.relYaw) > 10
  );

  // --------------------------------------------------------------------------
  // 1. TWO-TIER ANGLED-PHONE DETECTION PIPELINE
  // --------------------------------------------------------------------------
  for (const phone of rawPhoneCandidates) {
    const isLowerFrame = (phone.cy >= 0.35 * state.canvasHeight); // Lower 65% of screen (desk/lap level)

    // Contextual Posture Boost: +0.20 confidence boost when candidate glances down/sideways at desk level
    let effectiveScore = phone.rawScore;
    let contextBoosted = false;
    if (isLowerFrame && isCheatingPose && phone.rawScore >= 0.18 && phone.rawScore < 0.50) {
      effectiveScore = Math.min(0.95, phone.rawScore + 0.20);
      contextBoosted = true;
      phone.score = Math.round(effectiveScore * 100);
      phone.contextBoosted = true;
    }

    // 1.1 Base Confidence Check (discard below base 0.20 threshold)
    if (effectiveScore < state.config.phoneConf) {
      suppressedObjects.push({
        ...phone,
        reason: `LOW CONF (${phone.score}% < ${Math.round(state.config.phoneConf * 100)}%)`,
        type: "conf"
      });
      continue;
    }

    // 1.2 Aspect Ratio & Size Sanity Check
    if (!isValidPhoneGeometry(phone.box)) {
      suppressedObjects.push({
        ...phone,
        reason: "INVALID GEOMETRY (AREA/ASPECT)",
        type: "geometry"
      });
      continue;
    }

    // 1.3 Pre-calibrated Room Background Mask Check
    if (isMaskedByRoomCalibration(phone.box)) {
      suppressedObjects.push({
        ...phone,
        reason: "ROOM MASK CALIBRATED (PRE-EXISTING CLUTTER)",
        type: "mask"
      });
      continue;
    }

    // 1.4 Upper-Background & Clothes Hanger Ignore (Centroid Motion Lock: < 6px over 2.5s)
    if (state.config.useMotionLock) {
      const motion = checkObjectMotionLock(phone, now);
      const isOutsideReach = state.config.useWorkspaceROI && roi
        ? !isInsideWorkspaceROI(phone.box, roi)
        : (phone.cy < 0.40 * state.canvasHeight);

      if (motion.isStationary && isOutsideReach) {
        suppressedObjects.push({
          ...phone,
          reason: "STATIC CLOTHES / HANGER (<6px over 2.5s)",
          type: "motion"
        });
        continue;
      }
    }

    // Determine Tier based on effective confidence
    const isTier1 = (effectiveScore >= state.config.tier1Conf); // Tier 1 threshold: 0.50

    // ------------------------------------------------------------------------
    // TIER 1: High Confidence (>= 0.50) -> Accept directly inside Workspace ROI
    // ------------------------------------------------------------------------
    if (isTier1) {
      if (state.config.useWorkspaceROI && roi && !isInsideWorkspaceROI(phone.box, roi)) {
        suppressedObjects.push({
          ...phone,
          reason: "OUTSIDE WORKSPACE ROI (UPPER ROOM BACKGROUND)",
          type: "roi"
        });
        continue;
      }
      phone.tier = 1;
      activeProhibitedObjects.push(phone);
      continue;
    }

    // ------------------------------------------------------------------------
    // TIER 2: Angled / Low-Profile Phone Zone (0.20 - 0.49)
    // ------------------------------------------------------------------------
    // Do NOT reject outright. Pass candidate through Rigid Edge + Non-Skin Core Verification
    phone.tier = 2;
    const analysis = verifyAngledPhone(phone.box);
    phone.hasRigidEdge = analysis.hasRigidEdge;
    phone.hasPhoneCore = analysis.hasPhoneCore;
    phone.skinRatio = analysis.skinRatio;
    phone.coreRatio = analysis.coreRatio;

    // Decision Rule for Tier 2:
    // Only reject as a Bare Hand if skin > 80% AND both hasRigidEdge and hasPhoneCore are False
    if (
      state.config.useSkinFilter &&
      analysis.skinRatio > state.config.skinRatioThreshold &&
      !analysis.hasRigidEdge &&
      !analysis.hasPhoneCore
    ) {
      suppressedObjects.push({
        ...phone,
        reason: `BARE HAND REJECTED (${Math.round(analysis.skinRatio * 100)}% skin, no edge/core)`,
        type: "hand"
      });
      continue;
    }

    // Accept angled phone if (hasRigidEdge OR hasPhoneCore OR contextBoosted) is True
    if (analysis.hasRigidEdge || analysis.hasPhoneCore || contextBoosted) {
      activeProhibitedObjects.push(phone);
    } else {
      suppressedObjects.push({
        ...phone,
        reason: `UNVERIFIED LOW-CONF BLOB (${phone.score}%, no rigid edge/core)`,
        type: "conf"
      });
    }
  }

  // --------------------------------------------------------------------------
  // 2. FILTER BOOK DETECTIONS
  // --------------------------------------------------------------------------
  for (const book of rawBookCandidates) {
    if (book.rawScore < state.config.bookConf) {
      suppressedObjects.push({
        ...book,
        reason: `LOW CONF (${book.score}% < ${Math.round(state.config.bookConf * 100)}%)`,
        type: "conf"
      });
      continue;
    }

    if (state.config.useWorkspaceROI && roi && !isInsideWorkspaceROI(book.box, roi)) {
      suppressedObjects.push({
        ...book,
        reason: "OUTSIDE WORKSPACE ROI",
        type: "roi"
      });
      continue;
    }

    if (isMaskedByRoomCalibration(book.box)) {
      suppressedObjects.push({
        ...book,
        reason: "ROOM MASK (BACKGROUND)",
        type: "mask"
      });
      continue;
    }

    if (state.config.useMotionLock) {
      const motion = checkObjectMotionLock(book, now);
      if (motion.isStationary) {
        suppressedObjects.push({
          ...book,
          reason: "STATIC BACKGROUND (IGNORED)",
          type: "motion"
        });
        continue;
      }
    }

    activeProhibitedObjects.push(book);
  }

  // --------------------------------------------------------------------------
  // 3. FACE-VERIFIED MULTI-PERSON RULE (Clothes on Hanger Ignore)
  // --------------------------------------------------------------------------
  let isMultiPersonCandidate = false;
  if (numFaces > 1) {
    // MediaPipe confirmed > 1 distinct facial meshes
    isMultiPersonCandidate = true;
  } else if (rawPersonDetections.length > 1) {
    // Candidate has 1 face, but multiple person boxes detected (e.g. jackets on hangers)
    // Separate primary candidate person from secondary person boxes
    for (const p of rawPersonDetections) {
      const containsFace = doesBoxContainFace(p.box, faceResult);
      if (!containsFace) {
        // Secondary person candidate: check confidence and active motion
        const motion = checkObjectMotionLock(p, now);
        if (p.rawScore > 0.65 && !motion.isStationary && motion.maxShift > 15) {
          // High-confidence moving secondary person
          isMultiPersonCandidate = true;
        } else {
          // Stationary coat, hoodie, mannequin, or clothes hanger
          suppressedObjects.push({
            ...p,
            reason: "STATIC CLOTHES / HANGER (NO FACE, STATIONARY)",
            type: "hanger"
          });
        }
      }
    }
  }

  // --------------------------------------------------------------------------
  // 4. TEMPORAL PERSISTENCE (Rolling Buffer: 2 of 5 Frames + 1.5s Memory)
  // --------------------------------------------------------------------------
  const hasPhoneActive = activeProhibitedObjects.some(o => o.catName === "cell phone");
  const hasBookActive = activeProhibitedObjects.some(o => o.catName === "book");

  // Push to rolling history (max 5 frames)
  state.suppression.temporalHistory.push({
    hasPhone: hasPhoneActive,
    hasBook: hasBookActive,
    hasMultiPeople: isMultiPersonCandidate,
    timestamp: now
  });

  if (state.suppression.temporalHistory.length > 5) {
    state.suppression.temporalHistory.shift();
  }

  const history = state.suppression.temporalHistory;
  const phoneHits = history.filter(h => h.hasPhone).length;
  const bookHits = history.filter(h => h.hasBook).length;
  const multiHits = history.filter(h => h.hasMultiPeople).length;
  const reqFrames = state.config.temporalConfirmFrames; // default 2 of 5 frames

  let isPhoneConfirmed = phoneHits >= reqFrames;
  const isBookConfirmed = bookHits >= reqFrames;
  const isMultiPersonConfirmed = multiHits >= reqFrames;

  // 1.5s Bounding Box Memory Persistence
  if (isPhoneConfirmed && hasPhoneActive) {
    const primaryPhone = activeProhibitedObjects.find(o => o.catName === "cell phone");
    state.suppression.lastConfirmedMemory = {
      timestamp: now,
      item: primaryPhone ? { ...primaryPhone } : null
    };
  } else if (state.suppression.lastConfirmedMemory) {
    const memElapsed = now - state.suppression.lastConfirmedMemory.timestamp;
    if (memElapsed <= 1500) {
      // Memory active within 1.5s: bridge brief 1-frame occlusion drops
      isPhoneConfirmed = true;
      if (!hasPhoneActive && state.suppression.lastConfirmedMemory.item) {
        activeProhibitedObjects.push({
          ...state.suppression.lastConfirmedMemory.item,
          isMemoryBridged: true
        });
      }
    } else {
      state.suppression.lastConfirmedMemory = null;
    }
  }

  // Partition confirmed vs pending objects
  const finalActiveProhibited = [];
  for (const obj of activeProhibitedObjects) {
    if (obj.catName === "cell phone") {
      if (isPhoneConfirmed) {
        finalActiveProhibited.push(obj);
      } else {
        pendingConfirmationObjects.push({
          ...obj,
          confirmProgress: `${phoneHits}/${reqFrames}`
        });
      }
    } else if (obj.catName === "book") {
      if (isBookConfirmed) {
        finalActiveProhibited.push(obj);
      } else {
        pendingConfirmationObjects.push({
          ...obj,
          confirmProgress: `${bookHits}/${reqFrames}`
        });
      }
    } else {
      finalActiveProhibited.push(obj);
    }
  }

  return {
    activeProhibitedObjects: finalActiveProhibited,
    pendingConfirmationObjects,
    suppressedObjects,
    isMultiPersonConfirmed,
    multiPersonPendingCount: multiHits
  };
}

// ============================================================================
// STAGE 2: Two-Tier Angled-Phone Verification Check (Rigid Edge + Non-Skin Core)
// ============================================================================
function verifyAngledPhone(box) {
  const x = Math.max(0, Math.floor(box.x));
  const y = Math.max(0, Math.floor(box.y));
  const w = Math.min(state.canvasWidth - x, Math.floor(box.w));
  const h = Math.min(state.canvasHeight - y, Math.floor(box.h));

  if (w <= 8 || h <= 8) {
    return { hasRigidEdge: false, hasPhoneCore: false, skinRatio: 0, coreRatio: 0 };
  }

  try {
    const imgData = state.ctx.getImageData(x, y, w, h);
    const data = imgData.data;
    const totalPixels = w * h;

    const gray = new Uint8Array(totalPixels);
    let skinCount = 0;
    let darkOrGlareCount = 0;

    for (let py = 0; py < h; py++) {
      for (let px = 0; px < w; px++) {
        const idx = (py * w + px) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];

        // Grayscale conversion for edge detection
        gray[py * w + px] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);

        // YCrCb color space skin detection (Cr: 133-173, Cb: 77-127)
        const Cr = 0.500 * r - 0.419 * g - 0.081 * b + 128;
        const Cb = -0.169 * r - 0.331 * g + 0.500 * b + 128;

        const isSkin = (Cr >= 133 && Cr <= 173 && Cb >= 77 && Cb <= 127);
        if (isSkin) {
          skinCount++;
        } else {
          // Inspect non-skin region:
          // HSV Value channel V = max(r, g, b)
          const V = Math.max(r, g, b);
          // Dark metallic slab / phone chassis: V < 65
          // Emissive angled screen glow: V > 210
          if (V < 65 || V > 210) {
            darkOrGlareCount++;
          }
        }
      }
    }

    const skinRatio = skinCount / (totalPixels || 1);
    const coreRatio = darkOrGlareCount / (totalPixels || 1);
    const hasPhoneCore = (coreRatio >= 0.12);

    // -------------------------------------------------------------------------
    // Straight Geometric Edge Detection (Sobel + Hough Line Transform)
    // Bare hands and clothes have soft, curved contours. Tilted phones produce
    // crisp linear segments (> 35% of box dimension).
    // -------------------------------------------------------------------------
    const minDimension = Math.max(w, h);
    const minLineLen = Math.max(8, Math.floor(minDimension * 0.35));

    // Compute Sobel 3x3 gradients
    const edgePoints = [];
    for (let py = 1; py < h - 1; py++) {
      for (let px = 1; px < w - 1; px++) {
        const gx =
          -gray[(py - 1) * w + (px - 1)] + gray[(py - 1) * w + (px + 1)]
          - 2 * gray[py * w + (px - 1)] + 2 * gray[py * w + (px + 1)]
          - gray[(py + 1) * w + (px - 1)] + gray[(py + 1) * w + (px + 1)];

        const gy =
          -gray[(py - 1) * w + (px - 1)] - 2 * gray[(py - 1) * w + px] - gray[(py - 1) * w + (px + 1)]
          + gray[(py + 1) * w + (px - 1)] + 2 * gray[(py + 1) * w + px] + gray[(py + 1) * w + (px + 1)];

        const mag = Math.abs(gx) + Math.abs(gy);
        if (mag > 110) {
          edgePoints.push({ x: px, y: py });
        }
      }
    }

    let hasRigidEdge = false;
    if (edgePoints.length >= minLineLen) {
      // Hough Transform line accumulator with 36 angle bins (0° to 175° in 5° steps)
      const numThetas = 36;
      const diag = Math.ceil(Math.hypot(w, h));
      const rOffset = diag;
      const numR = diag * 2 + 1;
      const accumulator = new Uint16Array(numThetas * numR);

      const cosTable = new Float32Array(numThetas);
      const sinTable = new Float32Array(numThetas);
      for (let t = 0; t < numThetas; t++) {
        const rad = (t * 5 * Math.PI) / 180;
        cosTable[t] = Math.cos(rad);
        sinTable[t] = Math.sin(rad);
      }

      const step = edgePoints.length > 500 ? 2 : 1;
      const thresholdVotes = Math.max(5, Math.floor(minLineLen / step));

      for (let i = 0; i < edgePoints.length; i += step) {
        const pt = edgePoints[i];
        for (let t = 0; t < numThetas; t++) {
          const r = Math.round(pt.x * cosTable[t] + pt.y * sinTable[t]) + rOffset;
          if (r >= 0 && r < numR) {
            const accIdx = t * numR + r;
            accumulator[accIdx]++;
            if (accumulator[accIdx] >= thresholdVotes) {
              hasRigidEdge = true;
              break;
            }
          }
        }
        if (hasRigidEdge) break;
      }
    }

    return { hasRigidEdge, hasPhoneCore, skinRatio, coreRatio };
  } catch (err) {
    console.warn("verifyAngledPhone error:", err);
    return { hasRigidEdge: false, hasPhoneCore: false, skinRatio: 0, coreRatio: 0 };
  }
}

function isValidPhoneGeometry(box) {
  const frameArea = state.canvasWidth * state.canvasHeight;
  const boxArea = box.w * box.h;
  const areaRatio = boxArea / (frameArea || 1);

  // Reject if area < 1.5% or > 35% of total frame
  if (areaRatio < 0.015 || areaRatio > 0.35) return false;

  // Aspect ratio sanity check (reject extreme 10:1 slivers)
  const ar = box.w / (box.h || 1);
  if (ar > 4.0 || ar < 0.25) return false;

  return true;
}

// ============================================================================
// STAGE 1: Static Background Motion Lock & Centroid Tracking
// ============================================================================
function checkObjectMotionLock(item, now) {
  const cx = item.cx;
  const cy = item.cy;
  const category = item.catName;
  const trackedMap = state.suppression.trackedObjects;

  let matchedKey = null;
  let minDistance = 65; // px spatial grouping threshold

  for (const [key, track] of trackedMap.entries()) {
    if (track.category === category) {
      const dist = Math.hypot(cx - track.currentCentroid.x, cy - track.currentCentroid.y);
      if (dist < minDistance) {
        minDistance = dist;
        matchedKey = key;
      }
    }
  }

  if (matchedKey) {
    const track = trackedMap.get(matchedKey);
    const shiftFromInit = Math.hypot(cx - track.initialCentroid.x, cy - track.initialCentroid.y);
    track.maxShift = Math.max(track.maxShift, shiftFromInit);
    track.currentCentroid = { x: cx, y: cy };
    track.lastSeen = now;
    track.framesSeen++;

    // If present for >= 2.5s and centroid shifted <= 6px:
    const elapsedMs = now - track.firstSeen;
    if (elapsedMs >= 2500 && track.maxShift <= 6.0) {
      track.isStationary = true;
    }

    return { isStationary: track.isStationary, maxShift: track.maxShift };
  } else {
    // Register new tracked candidate
    const id = `${category}_${Math.round(cx)}_${Math.round(cy)}`;
    trackedMap.set(id, {
      category,
      initialCentroid: { x: cx, y: cy },
      currentCentroid: { x: cx, y: cy },
      firstSeen: now,
      lastSeen: now,
      maxShift: 0,
      framesSeen: 1,
      isStationary: false
    });

    // Prune stale entries older than 5.0 seconds
    if (trackedMap.size > 25) {
      for (const [key, t] of trackedMap.entries()) {
        if (now - t.lastSeen > 5000) {
          trackedMap.delete(key);
        }
      }
    }

    return { isStationary: false, maxShift: 0 };
  }
}

// ============================================================================
// STAGE 1: Active Workspace ROI (Candidate Reach Zone)
// ============================================================================
function getCandidateWorkspaceROI(faceLandmarks) {
  if (!faceLandmarks || faceLandmarks.length === 0) return null;

  let minX = 1.0, maxX = 0.0, minY = 1.0, maxY = 0.0;
  for (const pt of faceLandmarks) {
    if (pt.x < minX) minX = pt.x;
    if (pt.x > maxX) maxX = pt.x;
    if (pt.y < minY) minY = pt.y;
    if (pt.y > maxY) maxY = pt.y;
  }

  // Map to mirrored canvas coordinates
  const faceLeft = (1.0 - maxX) * state.canvasWidth;
  const faceRight = (1.0 - minX) * state.canvasWidth;
  const faceTop = minY * state.canvasHeight;
  const faceBottom = maxY * state.canvasHeight;

  const faceW = faceRight - faceLeft;
  const faceH = faceBottom - faceTop;
  const faceCenterX = (faceLeft + faceRight) / 2;

  // Active Workspace Reach Zone:
  // 2.2x to 3.8x shoulder width horizontally, from above head down to desk/lap
  const roiWidth = Math.max(faceW * 4.2, 420);
  const roiLeft = Math.max(0, faceCenterX - roiWidth / 2);
  const roiRight = Math.min(state.canvasWidth, faceCenterX + roiWidth / 2);
  const roiTop = Math.max(0, faceTop - faceH * 0.45);
  const roiBottom = Math.min(state.canvasHeight, faceBottom + faceH * 4.2);

  return {
    left: roiLeft,
    right: roiRight,
    top: roiTop,
    bottom: roiBottom,
    width: roiRight - roiLeft,
    height: roiBottom - roiTop
  };
}

function isInsideWorkspaceROI(box, roi) {
  if (!roi || !state.config.useWorkspaceROI) return true;
  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;
  return (cx >= roi.left && cx <= roi.right && cy >= roi.top && cy <= roi.bottom);
}

function doesBoxContainFace(box, faceResult) {
  if (!faceResult || !faceResult.faceLandmarks) return false;
  for (const landmarks of faceResult.faceLandmarks) {
    const nose = landmarks[1];
    const noseX = (1.0 - nose.x) * state.canvasWidth;
    const noseY = nose.y * state.canvasHeight;
    if (noseX >= box.x && noseX <= box.x + box.w && noseY >= box.y && noseY <= box.y + box.h) {
      return true;
    }
  }
  return false;
}

// ============================================================================
// STAGE 4: Room Background Mask Calibration
// ============================================================================
function startRoomMaskCapture() {
  if (state.suppression.isCapturingRoomMask) return;

  state.suppression.isCapturingRoomMask = true;
  state.suppression.roomMaskStartTime = performance.now();
  state.suppression.roomMaskSampledZones = [];

  audioSynth.playWarningChime();
  dom.roomMaskOverlay.classList.remove("hidden");
  dom.maskScanProgressFill.style.width = "0%";
  setHUDStatus("warning", "CALIBRATING ROOM MASK (2.0s)", "Hold still. Scanning static background clutter and hangers.");
}

function handleRoomMaskSampling(rawDetections, now) {
  const elapsed = now - state.suppression.roomMaskStartTime;
  const progressPercent = Math.min(100, (elapsed / state.suppression.roomMaskDuration) * 100);
  dom.maskScanProgressFill.style.width = `${progressPercent}%`;

  // Sample detected bounding boxes in background
  if (rawDetections && rawDetections.length > 0) {
    for (const det of rawDetections) {
      const b = det.boundingBox;
      const boxX = (1.0 - (b.originX + b.width) / state.canvasWidth) * state.canvasWidth;
      const boxY = b.originY;
      state.suppression.roomMaskSampledZones.push({
        x: boxX,
        y: boxY,
        w: b.width,
        h: b.height,
        cat: det.categories[0]?.categoryName || "clutter"
      });
    }
  }

  if (elapsed >= state.suppression.roomMaskDuration) {
    finalizeRoomMaskCapture();
  }
}

function finalizeRoomMaskCapture() {
  state.suppression.isCapturingRoomMask = false;
  dom.roomMaskOverlay.classList.add("hidden");

  // Cluster and merge sampled zones
  const sampled = state.suppression.roomMaskSampledZones;
  const mergedZones = [];

  for (const s of sampled) {
    const existing = mergedZones.find(m => Math.hypot((m.x + m.w / 2) - (s.x + s.w / 2), (m.y + m.h / 2) - (s.y + s.h / 2)) < 50);
    if (!existing) {
      mergedZones.push({ ...s });
    }
  }

  state.suppression.roomMaskZones = mergedZones;

  // Update UI Pills & Drawer
  dom.roomMaskActivePill.classList.remove("hidden");
  dom.roomMaskCount.textContent = `${mergedZones.length} ZONES`;
  dom.cfgRoomMaskSummary.textContent = `${mergedZones.length} background zones calibrated`;
  dom.btnClearRoomMask.disabled = (mergedZones.length === 0);

  audioSynth.playSuccessChime();
  setHUDStatus("normal", "ROOM MASK CALIBRATED", `${mergedZones.length} background zones masked from false positives`);
  logEvent("INFO", "ROOM_MASK", `Room background mask created: ${mergedZones.length} static clutter zones saved`, false);
}

function clearRoomMask() {
  state.suppression.roomMaskZones = [];
  dom.roomMaskActivePill.classList.add("hidden");
  dom.roomMaskCount.textContent = "0 ZONES";
  dom.cfgRoomMaskSummary.textContent = "0 zones calibrated";
  dom.btnClearRoomMask.disabled = true;

  logEvent("INFO", "ROOM_MASK_CLEARED", "Room background mask zones cleared", false);
}

function isMaskedByRoomCalibration(box) {
  const maskZones = state.suppression.roomMaskZones;
  if (!maskZones || maskZones.length === 0) return false;

  const cx = box.x + box.w / 2;
  const cy = box.y + box.h / 2;

  for (const zone of maskZones) {
    const zx = zone.x + zone.w / 2;
    const zy = zone.y + zone.h / 2;
    const dist = Math.hypot(cx - zx, cy - zy);

    // Centroid distance or IoU overlap
    if (dist < 45) return true;

    // Check intersection
    const interLeft = Math.max(box.x, zone.x);
    const interTop = Math.max(box.y, zone.y);
    const interRight = Math.min(box.x + box.w, zone.x + zone.w);
    const interBottom = Math.min(box.y + box.h, zone.y + zone.h);

    if (interRight > interLeft && interBottom > interTop) {
      const interArea = (interRight - interLeft) * (interBottom - interTop);
      const boxArea = box.w * box.h;
      if (interArea / boxArea > 0.35) return true;
    }
  }

  return false;
}

// ============================================================================
// Head Pose & Gaze Euler Angle Calculation (Matrix + Landmark Geometry)
// ============================================================================
function calculateHeadPose(landmarks, matrix) {
  const nose = landmarks[1];
  const chin = landmarks[152];
  const rightEye = landmarks[33];
  const leftEye = landmarks[263];

  let rawYaw = 0;
  let rawPitch = 0;
  let rawRoll = 0;

  if (matrix && matrix.data && matrix.data.length >= 16) {
    const d = matrix.data;
    const pitchM = Math.atan2(-d[6], Math.sqrt(d[0] * d[0] + d[4] * d[4])) * (180 / Math.PI);
    const yawM = Math.atan2(d[2], d[10]) * (180 / Math.PI);
    const rollM = Math.atan2(d[4], d[0]) * (180 / Math.PI);

    rawYaw = yawM;
    rawPitch = pitchM;
    rawRoll = rollM;
  }

  const eyeCenterX = (rightEye.x + leftEye.x) / 2;
  const eyeCenterY = (rightEye.y + leftEye.y) / 2;
  const eyeDist = Math.hypot(leftEye.x - rightEye.x, leftEye.y - rightEye.y);

  const noseOffsetX = (nose.x - eyeCenterX) / (eyeDist * 0.5);
  const geoYaw = -noseOffsetX * 48; // Inverted for mirrored view

  const verticalFaceSpan = chin.y - eyeCenterY;
  const noseOffsetY = (nose.y - eyeCenterY) / (verticalFaceSpan || 1);
  const geoPitch = (noseOffsetY - 0.42) * 90;

  const geoRoll = Math.atan2(leftEye.y - rightEye.y, leftEye.x - rightEye.x) * (180 / Math.PI);

  const currentYaw = (rawYaw !== 0) ? (rawYaw * 0.5 + geoYaw * 0.5) : geoYaw;
  const currentPitch = (rawPitch !== 0) ? (rawPitch * 0.5 + geoPitch * 0.5) : geoPitch;
  const currentRoll = (rawRoll !== 0) ? (rawRoll * 0.5 + geoRoll * 0.5) : geoRoll;

  // EMA smoothing
  const alpha = 0.35;
  state.pose.yaw = alpha * currentYaw + (1 - alpha) * state.pose.yaw;
  state.pose.pitch = alpha * currentPitch + (1 - alpha) * state.pose.pitch;
  state.pose.roll = alpha * currentRoll + (1 - alpha) * state.pose.roll;

  state.pose.relYaw = state.pose.yaw - state.calibration.baselineYaw;
  state.pose.relPitch = state.pose.pitch - state.calibration.baselinePitch;
  state.pose.relRoll = state.pose.roll - state.calibration.baselineRoll;

  const yThresh = state.config.yawThreshold;
  const pThresh = state.config.pitchThreshold;

  if (state.pose.relYaw > yThresh) {
    state.pose.direction = "LOOKING RIGHT";
  } else if (state.pose.relYaw < -yThresh) {
    state.pose.direction = "LOOKING LEFT";
  } else if (state.pose.relPitch > pThresh) {
    state.pose.direction = "LOOKING DOWN";
  } else if (state.pose.relPitch < -pThresh * 1.3) {
    state.pose.direction = "LOOKING UP";
  } else {
    state.pose.direction = "CENTERED";
  }
}

// ============================================================================
// Posture Neutral Calibration (Zero Baseline for Laptop Angle)
// ============================================================================
function calibrateNeutralPosture() {
  state.calibration.baselineYaw = state.pose.yaw;
  state.calibration.baselinePitch = state.pose.pitch;
  state.calibration.baselineRoll = state.pose.roll;
  state.calibration.calibrated = true;

  audioSynth.playSuccessChime();

  dom.postureStatusPill.textContent = "CALIBRATED (0° BASELINE)";
  dom.postureStatusPill.style.backgroundColor = "rgba(56, 189, 248, 0.2)";
  dom.postureStatusPill.style.color = "var(--color-primary)";

  setHUDStatus("normal", "POSTURE CALIBRATED", "Baseline zeroed to your natural screen reading angle");
  logEvent("INFO", "CALIBRATION", `Neutral posture calibrated: Yaw: ${state.pose.yaw.toFixed(1)}°, Pitch: ${state.pose.pitch.toFixed(1)}°`, false);

  const btn = dom.btnCalibrate;
  const origText = btn.innerHTML;
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg><span>Zeroed Successfully!</span>`;
  btn.classList.add("btn-primary");
  setTimeout(() => {
    btn.innerHTML = origText;
    btn.classList.remove("btn-primary");
  }, 2000);
}

// ============================================================================
// Core Exam Integrity Rules Engine
// ============================================================================
function evaluateIntegrityRules(
  numFaces,
  isMultiPersonConfirmed,
  multiPersonPendingCount,
  activeProhibitedObjects,
  pendingConfirmationObjects
) {
  const now = performance.now();
  let hasActiveViolation = false;

  // RULE 1: Candidate Absence (0 faces detected for > 2.0s)
  if (numFaces === 0) {
    if (!state.absenceStartTime) {
      state.absenceStartTime = now;
    }
    const absenceElapsed = (now - state.absenceStartTime) / 1000;

    if (absenceElapsed >= state.config.absenceTime) {
      hasActiveViolation = true;
      setHUDStatus("violation", "ABSENT FROM FRAME", `No candidate detected for ${absenceElapsed.toFixed(1)}s`);
      
      if (!state.sustainedAbsenceTriggered || canTriggerViolation("ABSENT")) {
        state.sustainedAbsenceTriggered = true;
        triggerViolation(
          "ABSENT",
          "HIGH",
          "Candidate Absent from Frame",
          `Zero faces detected for ${absenceElapsed.toFixed(1)}s (exceeded ${state.config.absenceTime}s threshold)`,
          10
        );
      }
    } else {
      setHUDStatus("warning", "FACE NOT DETECTED", `Absence timer: ${absenceElapsed.toFixed(1)}s / ${state.config.absenceTime}s`);
    }
  } else {
    if (state.absenceStartTime) {
      const totalAbsence = (now - state.absenceStartTime) / 1000;
      if (totalAbsence >= 1.0) {
        state.totalAbsenceDuration += totalAbsence;
        dom.kpiAbsenceDuration.textContent = `${state.totalAbsenceDuration.toFixed(1)}s`;
      }
      state.absenceStartTime = null;
      state.sustainedAbsenceTriggered = false;
    }
  }

  // RULE 2: Multiple People Detected (Face-Verified & Confirmed)
  if (isMultiPersonConfirmed) {
    hasActiveViolation = true;
    setHUDStatus("violation", "MULTIPLE PEOPLE DETECTED", "Verified unauthorized presence detected in exam perimeter");

    if (canTriggerViolation("MULTIPLE_PEOPLE")) {
      triggerViolation(
        "MULTIPLE_PEOPLE",
        "HIGH",
        "Multiple People Confirmed",
        "Face-verified multiple persons active in exam perimeter",
        20
      );
    }
  } else if (multiPersonPendingCount > 0 && multiPersonPendingCount < state.config.temporalConfirmFrames) {
    setHUDStatus("warning", "ANALYZING MULTI-PERSON (PENDING)", `Confirming movement (${multiPersonPendingCount}/${state.config.temporalConfirmFrames} frames)...`);
  }

  // RULE 3: Prohibited Devices & Objects (Confirmed via Temporal Buffer)
  if (activeProhibitedObjects.length > 0) {
    hasActiveViolation = true;
    const item = activeProhibitedObjects[0];
    setHUDStatus("violation", `PROHIBITED ITEM: ${item.name.toUpperCase()}`, `Confirmed with ${item.score}% confidence`);

    if (canTriggerViolation("PROHIBITED_OBJECT")) {
      state.devicesFlaggedCount += activeProhibitedObjects.length;
      dom.kpiDevicesCount.textContent = state.devicesFlaggedCount;

      triggerViolation(
        "PROHIBITED_OBJECT",
        "HIGH",
        `Prohibited Device: ${item.name}`,
        `Unauthorized object verified in frame: ${item.name} (${item.score}% confidence)`,
        15
      );
    }
  } else if (pendingConfirmationObjects.length > 0) {
    const item = pendingConfirmationObjects[0];
    setHUDStatus("warning", `ANALYZING ITEM: ${item.name.toUpperCase()}`, `Confirming persistence (${item.confirmProgress} frames)...`);
  }

  // RULE 4: Sustained Look-Away Tracking (Left, Right, Down for > 3.0s)
  if (numFaces > 0 && !hasActiveViolation) {
    const isLookingAway = (state.pose.direction !== "CENTERED");

    if (isLookingAway) {
      if (!state.lookAwayStartTime) {
        state.lookAwayStartTime = now;
      }
      const lookAwayElapsed = (now - state.lookAwayStartTime) / 1000;
      const progressPercent = Math.min(100, (lookAwayElapsed / state.config.lookAwayTime) * 100);

      dom.lookawayBadge.classList.remove("hidden");
      dom.lookawayCountdown.textContent = `${lookAwayElapsed.toFixed(1)}s`;
      dom.lookawayProgressFill.style.width = `${progressPercent}%`;

      if (lookAwayElapsed >= state.config.lookAwayTime) {
        setHUDStatus("warning", `SUSTAINED LOOK-AWAY: ${state.pose.direction}`, `Continuous deviation for ${lookAwayElapsed.toFixed(1)}s`);

        if (!state.sustainedLookAwayTriggered || canTriggerViolation("LOOK_AWAY")) {
          state.sustainedLookAwayTriggered = true;
          triggerViolation(
            "LOOK_AWAY",
            "WARN",
            `Sustained Look-Away (${state.pose.direction})`,
            `Candidate deviated gaze (${state.pose.direction}) for ${lookAwayElapsed.toFixed(1)}s (Threshold: ${state.config.lookAwayTime}s)`,
            5
          );
        }
      } else {
        setHUDStatus("warning", `GAZE DEVIATION: ${state.pose.direction}`, `Look-away timer: ${lookAwayElapsed.toFixed(1)}s / ${state.config.lookAwayTime}s`);
      }
    } else {
      if (state.lookAwayStartTime) {
        const totalDeviated = (now - state.lookAwayStartTime) / 1000;
        if (totalDeviated >= 0.8) {
          state.totalLookAwayDuration += totalDeviated;
          dom.kpiLookAwayDuration.textContent = `${state.totalLookAwayDuration.toFixed(1)}s`;
        }
        state.lookAwayStartTime = null;
        state.sustainedLookAwayTriggered = false;
        dom.lookawayBadge.classList.add("hidden");
        dom.lookawayProgressFill.style.width = "0%";
      }

      setHUDStatus("normal", "NORMAL - CANDIDATE ATTENTIVE", "Candidate looking at screen center");
    }
  }

  updateCooldownUI();
}

// ============================================================================
// Violation Dispatcher & Cooldown Management
// ============================================================================
function canTriggerViolation(type) {
  const now = performance.now();
  const lastTrigger = state.cooldowns[type] || 0;
  return (now - lastTrigger) >= 5000;
}

function triggerViolation(type, severity, title, details, deduction) {
  state.cooldowns[type] = performance.now();

  state.integrityScore = Math.max(0, state.integrityScore - deduction);
  state.totalWarnings++;

  dom.kpiTotalWarnings.textContent = state.totalWarnings;
  updateIntegrityScoreUI();

  if (severity === "HIGH") {
    audioSynth.playViolationAlarm();
  } else {
    audioSynth.playWarningChime();
  }

  let evidenceThumb = null;
  if (state.config.snapshotsEnabled) {
    evidenceThumb = captureEvidenceSnapshot(title, details, severity);
  }

  logEvent(severity, title, details, !!evidenceThumb, evidenceThumb);

  if (state.backendConnected) {
    syncEventToCloud({
      sessionId: state.sessionId,
      candidate: state.candidate,
      type,
      severity,
      title,
      details,
      integrityScore: state.integrityScore,
      evidenceThumb
    });
  }
}

function updateCooldownUI() {
  const now = performance.now();
  let anyActive = false;
  for (const key in state.cooldowns) {
    if ((now - state.cooldowns[key]) < 5000) {
      anyActive = true;
      break;
    }
  }

  if (anyActive) {
    dom.cooldownIndicator.classList.add("cooldown-active");
    dom.cooldownText.textContent = "Violation Cooldown: Active (5s protection)";
  } else {
    dom.cooldownIndicator.classList.remove("cooldown-active");
    dom.cooldownText.textContent = "Violation Cooldown: Ready";
  }
}

// ============================================================================
// Visual Evidence Snapshot Capture (Canvas Frame + Security Watermark)
// ============================================================================
function captureEvidenceSnapshot(title, details, severity) {
  const offscreen = document.createElement("canvas");
  offscreen.width = state.canvasWidth;
  offscreen.height = state.canvasHeight;
  const offCtx = offscreen.getContext("2d");

  offCtx.drawImage(dom.canvas, 0, 0);

  const bannerColor = (severity === "HIGH") ? "rgba(244, 63, 94, 0.88)" : "rgba(245, 158, 11, 0.88)";
  offCtx.fillStyle = bannerColor;
  offCtx.fillRect(0, 0, offscreen.width, 42);

  offCtx.fillStyle = "#ffffff";
  offCtx.font = "bold 15px 'Inter', sans-serif";
  offCtx.fillText(`[EVIDENCE ARCHIVE] ${title.toUpperCase()}`, 16, 26);

  offCtx.fillStyle = "rgba(10, 14, 23, 0.85)";
  offCtx.fillRect(0, offscreen.height - 32, offscreen.width, 32);

  const timestamp = new Date().toLocaleTimeString();
  const dateStr = new Date().toISOString().split("T")[0];
  offCtx.font = "12px 'JetBrains Mono', monospace";
  offCtx.fillStyle = "#38bdf8";
  offCtx.fillText(`SENTINEL-AI SECURE AUDIT | ID: ${state.candidate.id} | ${dateStr} ${timestamp} | SCORE: ${state.integrityScore}%`, 16, offscreen.height - 11);

  const dataUrl = offscreen.toDataURL("image/jpeg", 0.82);

  const evidenceRecord = {
    id: "EV-" + Date.now().toString(36).toUpperCase(),
    timestamp: new Date().toLocaleTimeString(),
    date: dateStr,
    elapsed: dom.sessionTimer.textContent,
    title,
    details,
    severity,
    score: state.integrityScore,
    imageData: dataUrl
  };

  state.evidenceSnapshots.push(evidenceRecord);
  return evidenceRecord;
}

// ============================================================================
// Anti-Cheat: Tab & Focus Switching Detection
// ============================================================================
function setupAntiCheatListeners() {
  document.addEventListener("visibilitychange", () => {
    if (!state.sessionActive || !state.config.antiCheatTab) return;

    if (document.hidden) {
      state.tabSwitchesCount++;
      dom.kpiTabSwitches.textContent = state.tabSwitchesCount;

      triggerViolation(
        "TAB_SWITCH",
        "HIGH",
        "Exam Tab Switched / Minimized",
        "Candidate navigated away from the proctored exam tab or minimized browser",
        10
      );
    } else {
      logEvent("INFO", "FOCUS_RESTORED", "Candidate returned to the proctored exam tab", false);
    }
  });

  window.addEventListener("blur", () => {
    if (!state.sessionActive || !state.config.antiCheatTab || document.hidden) return;

    state.tabSwitchesCount++;
    dom.kpiTabSwitches.textContent = state.tabSwitchesCount;

    triggerViolation(
      "TAB_SWITCH",
      "WARN",
      "Window Focus Lost",
      "Candidate clicked outside of the exam window (secondary monitor or app switched)",
      6
    );
  });
}

// ============================================================================
// Canvas HUD Graphic Drawing (Mesh, Bounding Boxes, ROI, Suppressed Items)
// ============================================================================
function drawFaceLandmarksCustom(landmarks) {
  const ctx = state.ctx;
  ctx.save();
  ctx.fillStyle = "rgba(56, 189, 248, 0.4)";
  
  const anchorIndices = [1, 33, 263, 61, 291, 199, 10, 152];
  for (const idx of anchorIndices) {
    const pt = landmarks[idx];
    if (!pt) continue;
    const x = (1.0 - pt.x) * state.canvasWidth;
    const y = pt.y * state.canvasHeight;

    ctx.beginPath();
    ctx.arc(x, y, 3, 0, 2 * Math.PI);
    ctx.fill();
  }

  ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
  ctx.lineWidth = 1.5;
  const leftEyeIndices = [33, 160, 158, 133, 153, 144, 33];
  ctx.beginPath();
  for (let i = 0; i < leftEyeIndices.length; i++) {
    const pt = landmarks[leftEyeIndices[i]];
    const x = (1.0 - pt.x) * state.canvasWidth;
    const y = pt.y * state.canvasHeight;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  const rightEyeIndices = [362, 385, 387, 263, 373, 380, 362];
  ctx.beginPath();
  for (let i = 0; i < rightEyeIndices.length; i++) {
    const pt = landmarks[rightEyeIndices[i]];
    const x = (1.0 - pt.x) * state.canvasWidth;
    const y = pt.y * state.canvasHeight;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();

  ctx.restore();
}

function drawWorkspaceROI(roi) {
  const ctx = state.ctx;
  ctx.save();
  ctx.strokeStyle = "rgba(56, 189, 248, 0.45)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.strokeRect(roi.left, roi.top, roi.width, roi.height);

  ctx.fillStyle = "rgba(56, 189, 248, 0.65)";
  ctx.font = "bold 10px 'JetBrains Mono', monospace";
  ctx.fillText("ACTIVE WORKSPACE REACH ROI", roi.left + 8, roi.top + 14);
  ctx.restore();
}

function drawSuppressedObjects(suppressedList) {
  const ctx = state.ctx;
  ctx.save();

  for (const item of suppressedList) {
    const b = item.box;
    ctx.setLineDash([4, 4]);

    if (item.type === "hand") {
      ctx.strokeStyle = "rgba(245, 158, 11, 0.6)";
      ctx.fillStyle = "rgba(245, 158, 11, 0.75)";
    } else {
      ctx.strokeStyle = "rgba(148, 163, 184, 0.5)";
      ctx.fillStyle = "rgba(148, 163, 184, 0.75)";
    }

    ctx.lineWidth = 1.5;
    ctx.strokeRect(b.x, b.y, b.w, b.h);

    ctx.font = "9px 'JetBrains Mono', monospace";
    ctx.fillText(`[SUPPRESSED] ${item.reason}`, b.x + 4, Math.max(12, b.y - 4));
  }

  ctx.restore();
}

function drawProhibitedObjects(confirmedObjects, pendingObjects) {
  const ctx = state.ctx;
  ctx.save();

  // Draw Confirmed Violations in Neon Rose
  for (const obj of confirmedObjects) {
    const b = obj.box;
    ctx.strokeStyle = "rgba(244, 63, 94, 0.95)";
    ctx.lineWidth = 2.5;
    ctx.shadowColor = "#f43f5e";
    ctx.shadowBlur = 12;

    ctx.strokeRect(b.x, b.y, b.w, b.h);

    ctx.fillStyle = "rgba(244, 63, 94, 0.95)";
    ctx.fillRect(b.x, Math.max(0, b.y - 26), b.w, 26);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 12px 'JetBrains Mono', monospace";
    const label = obj.tier === 2
      ? `⚠️ ${obj.name.toUpperCase()} [TIER-2 ANGLED] (${obj.score}%)`
      : `⚠️ ${obj.name.toUpperCase()} (${obj.score}%)`;
    ctx.fillText(label, b.x + 6, Math.max(16, b.y - 8));
  }

  // Draw Pending Confirmation Items in Warning Amber
  for (const obj of pendingObjects) {
    const b = obj.box;
    ctx.strokeStyle = "rgba(245, 158, 11, 0.85)";
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 3]);

    ctx.strokeRect(b.x, b.y, b.w, b.h);

    ctx.fillStyle = "rgba(245, 158, 11, 0.85)";
    ctx.fillRect(b.x, Math.max(0, b.y - 22), b.w, 22);

    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 11px 'JetBrains Mono', monospace";
    ctx.fillText(`ANALYZING (${obj.confirmProgress})`, b.x + 4, Math.max(14, b.y - 6));
  }

  ctx.restore();
}

function drawHUDTargetReticle(landmarks) {
  const ctx = state.ctx;
  ctx.save();

  const nose = landmarks[1];
  const cx = (1.0 - nose.x) * state.canvasWidth;
  const cy = nose.y * state.canvasHeight;

  const pointerLength = 55;
  const radYaw = (state.pose.relYaw * Math.PI) / 180;
  const radPitch = (state.pose.relPitch * Math.PI) / 180;

  const px = cx + Math.sin(radYaw) * pointerLength;
  const py = cy + Math.sin(radPitch) * pointerLength;

  ctx.strokeStyle = (state.pose.direction === "CENTERED") ? "rgba(56, 189, 248, 0.8)" : "rgba(245, 158, 11, 0.9)";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(cx, cy);
  ctx.lineTo(px, py);
  ctx.stroke();

  ctx.fillStyle = (state.pose.direction === "CENTERED") ? "#38bdf8" : "#fbbf24";
  ctx.beginPath();
  ctx.arc(px, py, 4, 0, 2 * Math.PI);
  ctx.fill();

  ctx.restore();
}

// ============================================================================
// UI Updates: Gauges, Banners, Score, Incident Log
// ============================================================================
function updateAngleTelemetryUI() {
  const relY = state.pose.relYaw;
  const relP = state.pose.relPitch;
  const relR = state.pose.relRoll;

  dom.yawVal.textContent = `${relY > 0 ? "+" : ""}${relY.toFixed(1)}°`;
  dom.pitchVal.textContent = `${relP > 0 ? "+" : ""}${relP.toFixed(1)}°`;
  dom.rollVal.textContent = `${relR > 0 ? "+" : ""}${relR.toFixed(1)}°`;

  const yawPercent = Math.max(0, Math.min(100, ((relY + 45) / 90) * 100));
  dom.yawNeedle.style.left = `${yawPercent}%`;

  const pitchPercent = Math.max(0, Math.min(100, ((relP + 35) / 70) * 100));
  dom.pitchNeedle.style.left = `${pitchPercent}%`;

  const rollPercent = Math.max(0, Math.min(100, ((relR + 30) / 60) * 100));
  dom.rollNeedle.style.left = `${rollPercent}%`;

  dom.hudDirectionBadge.textContent = state.pose.direction;
  if (state.pose.direction === "CENTERED") {
    dom.hudDirectionBadge.style.color = "var(--color-emerald)";
  } else {
    dom.hudDirectionBadge.style.color = "var(--color-amber)";
  }
}

function setHUDStatus(type, title, subtitle) {
  dom.hudStatusBanner.className = `hud-status-banner status-${type}`;
  dom.hudStatusText.textContent = title;
  dom.hudStatusSubtext.textContent = subtitle;
}

function updateIntegrityScoreUI() {
  const score = state.integrityScore;
  dom.integrityScoreVal.textContent = score;

  const offset = 314.16 - (314.16 * score) / 100;
  dom.scoreCircleProgress.style.strokeDashoffset = offset;

  if (score >= 85) {
    dom.scoreCircleProgress.style.stroke = "var(--color-emerald)";
    dom.integrityVerdictBadge.textContent = "EXCELLENT";
    dom.integrityVerdictBadge.className = "score-badge badge-clean";
    dom.integrityDescription.textContent = "Session proceeding cleanly with high adherence to integrity standards.";
  } else if (score >= 65) {
    dom.scoreCircleProgress.style.stroke = "var(--color-amber)";
    dom.integrityVerdictBadge.textContent = "MODERATE CAUTION";
    dom.integrityVerdictBadge.className = "score-badge badge-caution";
    dom.integrityDescription.textContent = "Several warnings flagged. Candidate should maintain steady gaze on screen.";
  } else {
    dom.scoreCircleProgress.style.stroke = "var(--color-rose)";
    dom.integrityVerdictBadge.textContent = "HIGH RISK / BREACH";
    dom.integrityVerdictBadge.className = "score-badge badge-breach";
    dom.integrityDescription.textContent = "Significant integrity violations detected. Comprehensive audit report recommended.";
  }
}

function logEvent(severity, title, details, hasEvidence, evidenceRecord = null) {
  const timeStr = new Date().toLocaleTimeString();
  const elapsedStr = dom.sessionTimer.textContent;

  const eventItem = {
    index: state.incidentLog.length + 1,
    time: timeStr,
    elapsed: elapsedStr,
    severity,
    title,
    details,
    score: state.integrityScore,
    evidence: evidenceRecord
  };

  state.incidentLog.unshift(eventItem);
  dom.emptyLogState.classList.add("hidden");

  renderIncidentLogDOM(eventItem);
  updateFilterCounters();
}

function renderIncidentLogDOM(item) {
  const el = document.createElement("div");
  const severityClass = (item.severity === "HIGH") ? "severity-high" : (item.severity === "WARN" ? "severity-warning" : "severity-info");
  const tagClass = (item.severity === "HIGH") ? "high" : (item.severity === "WARN" ? "warn" : "info");

  el.className = `log-item ${severityClass}`;
  el.dataset.severity = item.severity.toLowerCase();
  el.dataset.hasEvidence = item.evidence ? "true" : "false";

  let thumbHtml = "";
  if (item.evidence && item.evidence.imageData) {
    thumbHtml = `
      <div class="log-thumb-wrapper" title="Click to inspect visual evidence">
        <img src="${item.evidence.imageData}" alt="Evidence Thumbnail">
        <div class="log-thumb-zoom">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
            <line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>
          </svg>
        </div>
      </div>
    `;
  }

  el.innerHTML = `
    <div class="log-item-left">
      ${thumbHtml}
      <div class="log-meta-block">
        <div class="log-item-header">
          <span class="log-tag ${tagClass}">${item.severity}</span>
          <span class="log-title">${item.title}</span>
        </div>
        <span class="log-desc">${item.details}</span>
      </div>
    </div>
    <div class="log-item-right">
      <span class="log-time mono">${item.time}</span>
      <span class="log-deduct mono">${item.severity === "HIGH" ? "-15 pts" : (item.severity === "WARN" ? "-5 pts" : "")}</span>
    </div>
  `;

  if (item.evidence) {
    const thumbWrap = el.querySelector(".log-thumb-wrapper");
    if (thumbWrap) {
      thumbWrap.addEventListener("click", () => openEvidenceModal(item.evidence));
    }
  }

  dom.logItemsList.insertBefore(el, dom.logItemsList.firstChild);
}

function updateFilterCounters() {
  const allCount = state.incidentLog.length;
  const highCount = state.incidentLog.filter(e => e.severity === "HIGH").length;
  const warnCount = state.incidentLog.filter(e => e.severity === "WARN").length;
  const evCount = state.incidentLog.filter(e => e.evidence !== null).length;

  dom.countFilterAll.textContent = allCount;
  dom.countFilterHigh.textContent = highCount;
  dom.countFilterWarn.textContent = warnCount;
  dom.countFilterEvidence.textContent = evCount;
}

function filterIncidentLog(filter) {
  const items = dom.logItemsList.querySelectorAll(".log-item");
  items.forEach(el => {
    const sev = el.dataset.severity;
    const hasEv = el.dataset.hasEvidence === "true";

    if (filter === "all") {
      el.style.display = "flex";
    } else if (filter === "high" && sev === "high") {
      el.style.display = "flex";
    } else if (filter === "warn" && sev === "warn") {
      el.style.display = "flex";
    } else if (filter === "evidence" && hasEv) {
      el.style.display = "flex";
    } else {
      el.style.display = "none";
    }
  });
}

// ============================================================================
// Session Controls: Start, Timer, End Exam
// ============================================================================
function startProctoredSession() {
  if (state.sessionActive) return;

  state.sessionActive = true;
  state.sessionStartTime = Date.now();
  dom.btnStartSession.disabled = true;
  dom.btnEndSession.disabled = false;
  dom.btnStartSessionText.textContent = "Proctoring In Progress";
  dom.recDot.style.display = "inline-block";

  audioSynth.playSuccessChime();

  state.integrityScore = 100;
  state.totalWarnings = 0;
  state.totalLookAwayDuration = 0;
  state.totalAbsenceDuration = 0;
  state.tabSwitchesCount = 0;
  state.devicesFlaggedCount = 0;
  updateIntegrityScoreUI();

  state.sessionTimerInterval = setInterval(() => {
    const elapsedMs = Date.now() - state.sessionStartTime;
    state.sessionElapsedTime = elapsedMs;
    const totalSec = Math.floor(elapsedMs / 1000);
    const hrs = String(Math.floor(totalSec / 3600)).padStart(2, "0");
    const mins = String(Math.floor((totalSec % 3600) / 60)).padStart(2, "0");
    const secs = String(totalSec % 60).padStart(2, "0");
    dom.sessionTimer.textContent = `${hrs}:${mins}:${secs}`;
  }, 1000);

  setHUDStatus("normal", "PROCTORING SESSION ACTIVE", "Real-time AI integrity monitoring active");
  logEvent("INFO", "SESSION_START", `Proctored session started for candidate ${state.candidate.name} (${state.candidate.id})`, false);

  if (state.backendConnected) {
    syncSessionStartToCloud();
  }
}

function endProctoredSession() {
  if (!state.sessionActive) return;

  state.sessionActive = false;
  clearInterval(state.sessionTimerInterval);
  dom.btnStartSession.disabled = false;
  dom.btnEndSession.disabled = true;
  dom.btnStartSessionText.textContent = "Start New Session";
  dom.recDot.style.display = "none";

  setHUDStatus("normal", "EXAM COMPLETED", "Session finalized. Review telemetry or download reports.");
  logEvent("INFO", "SESSION_END", `Exam session ended. Final Integrity Score: ${state.integrityScore}%`, false);

  generatePrintableReport();
}

// ============================================================================
// Visual Evidence Modal Preview
// ============================================================================
function openEvidenceModal(record) {
  state.selectedEvidence = record;
  dom.modalEvidenceTitle.textContent = record.title;
  dom.modalEvidenceImage.src = record.imageData;
  dom.modalEvidenceTime.textContent = `${record.date} ${record.timestamp}`;
  dom.modalEvidenceElapsed.textContent = record.elapsed;
  dom.modalEvidenceScore.textContent = `${record.score}%`;
  dom.modalEvidenceDetails.textContent = record.details;

  dom.evidenceModalBackdrop.classList.remove("hidden");
}

function closeEvidenceModal() {
  dom.evidenceModalBackdrop.classList.add("hidden");
  state.selectedEvidence = null;
}

function downloadSelectedEvidenceImage() {
  if (!state.selectedEvidence) return;
  const a = document.createElement("a");
  a.href = state.selectedEvidence.imageData;
  a.download = `Evidence_${state.selectedEvidence.id}_${state.candidate.id}.jpg`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

// ============================================================================
// 1-Click Exporting: CSV Audit Log & Printable PDF / Summary Report
// ============================================================================
function exportCSVAuditLog() {
  const headers = [
    "Event Index",
    "Session ID",
    "Candidate Name",
    "Candidate ID",
    "Exam Title",
    "Timestamp",
    "Elapsed Time",
    "Severity",
    "Violation Title",
    "Details",
    "Integrity Score After Deduction",
    "Evidence Snapshot Available"
  ];

  const rows = state.incidentLog.map(item => [
    item.index,
    `"${state.sessionId}"`,
    `"${state.candidate.name}"`,
    `"${state.candidate.id}"`,
    `"${state.candidate.examTitle}"`,
    `"${item.time}"`,
    `"${item.elapsed}"`,
    `"${item.severity}"`,
    `"${item.title.replace(/"/g, '""')}"`,
    `"${item.details.replace(/"/g, '""')}"`,
    item.score,
    item.evidence ? "YES" : "NO"
  ]);

  const csvContent = [headers.join(","), ...rows.map(r => r.join(","))].join("\r\n");
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.setAttribute("href", url);
  link.setAttribute("download", `Proctor_Audit_${state.candidate.id}_${state.sessionId}.csv`);
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
}

function generatePrintableReport() {
  const dateStr = new Date().toLocaleDateString();
  const timeStr = new Date().toLocaleTimeString();
  const isPass = state.integrityScore >= 65;
  const badgeClass = isPass ? "pass" : "flagged";
  const badgeText = isPass ? "PASSED & VERIFIED" : "FLAGGED FOR HUMAN REVIEW";

  let evidenceHtml = "";
  if (state.evidenceSnapshots.length > 0) {
    evidenceHtml = `
      <h3 style="margin-top:20px; font-size:12pt; border-bottom:1px solid #cbd5e1; padding-bottom:5px;">Photographic Evidence Archive</h3>
      <div class="report-evidence-grid">
        ${state.evidenceSnapshots.slice(0, 6).map(ev => `
          <div class="report-evidence-item">
            <img src="${ev.imageData}" alt="Evidence">
            <div class="report-evidence-title">${ev.title}</div>
            <div class="report-evidence-time">${ev.date} ${ev.timestamp} (Elapsed: ${ev.elapsed})</div>
          </div>
        `).join("")}
      </div>
    `;
  }

  dom.printReportContainer.innerHTML = `
    <div class="report-header">
      <div class="report-brand">
        <h1>SENTINEL-AI PROCTORING AUDIT</h1>
        <p>Comprehensive Client-Side Computer Vision Integrity Certificate</p>
      </div>
      <div class="report-badge ${badgeClass}">${badgeText}</div>
    </div>

    <div class="report-meta-grid">
      <div class="report-meta-row"><strong>Candidate:</strong> ${state.candidate.name} (ID: ${state.candidate.id})</div>
      <div class="report-meta-row"><strong>Exam:</strong> ${state.candidate.examTitle}</div>
      <div class="report-meta-row"><strong>Session ID:</strong> ${state.sessionId}</div>
      <div class="report-meta-row"><strong>Date / Time:</strong> ${dateStr} ${timeStr}</div>
      <div class="report-meta-row"><strong>Exam Duration:</strong> ${dom.sessionTimer.textContent}</div>
      <div class="report-meta-row"><strong>Suppression Filters:</strong> 4-Stage YCrCb Skin & Motion-Locked Active</div>
    </div>

    <div class="report-score-box">
      <div style="font-size:11pt; color:#64748b; font-weight:700;">FINAL INTEGRITY CONFIDENCE SCORE</div>
      <div class="report-score-num">${state.integrityScore}%</div>
      <div style="font-size:10pt; color:${isPass ? '#166534' : '#9f1239'}; font-weight:700;">
        ${isPass ? "VERDICT: ACCEPTABLE INTEGRITY CONFIRMED" : "VERDICT: CRITICAL SUSPECTED INTEGRITY INFRACTIONS"}
      </div>
    </div>

    <table class="report-kpi-table">
      <thead>
        <tr>
          <th>Metric Parameter</th>
          <th>Recorded Telemetry Value</th>
          <th>Threshold Rule Status</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Total Warnings Flagged</td>
          <td>${state.totalWarnings}</td>
          <td>${state.totalWarnings > 4 ? "Flagged (>4)" : "Normal"}</td>
        </tr>
        <tr>
          <td>Cumulative Look-Away Duration</td>
          <td>${state.totalLookAwayDuration.toFixed(1)} seconds</td>
          <td>Triggered sustained alert after ${state.config.lookAwayTime}s continuous hold</td>
        </tr>
        <tr>
          <td>Cumulative Absence Duration</td>
          <td>${state.totalAbsenceDuration.toFixed(1)} seconds</td>
          <td>Triggered absent alert after ${state.config.absenceTime}s continuous hold</td>
        </tr>
        <tr>
          <td>Browser Tab / Focus Switches</td>
          <td>${state.tabSwitchesCount} instances</td>
          <td>${state.tabSwitchesCount > 0 ? "Potential External Resource Access" : "Zero Tab Switches"}</td>
        </tr>
        <tr>
          <td>Prohibited Devices In Frame</td>
          <td>${state.devicesFlaggedCount} detected</td>
          <td>${state.devicesFlaggedCount > 0 ? "Unauthorized Hardware Confirmed" : "Zero Devices"}</td>
        </tr>
      </tbody>
    </table>

    <h3 style="margin-top:20px; font-size:12pt; border-bottom:1px solid #cbd5e1; padding-bottom:5px;">Incident Timeline Log (${state.incidentLog.length} events)</h3>
    <table class="report-kpi-table" style="margin-top:10px;">
      <thead>
        <tr>
          <th>#</th>
          <th>Time</th>
          <th>Elapsed</th>
          <th>Severity</th>
          <th>Incident Title</th>
          <th>Details</th>
          <th>Score</th>
        </tr>
      </thead>
      <tbody>
        ${state.incidentLog.slice(0, 15).map(item => `
          <tr>
            <td>${item.index}</td>
            <td>${item.time}</td>
            <td>${item.elapsed}</td>
            <td style="color:${item.severity === 'HIGH' ? '#9f1239' : (item.severity === 'WARN' ? '#b45309' : '#0369a1')}; font-weight:700;">
              ${item.severity}
            </td>
            <td><strong>${item.title}</strong></td>
            <td>${item.details}</td>
            <td>${item.score}%</td>
          </tr>
        `).join("")}
      </tbody>
    </table>

    ${evidenceHtml}

    <div class="report-footer">
      <div>Generated automatically by Sentinel-AI Proctor Engine. Certified cryptographic timestamp.</div>
      <div>Official Audit Copy &bull; Page 1 of 1</div>
    </div>
  `;

  window.print();
}

// ============================================================================
// Cloud Backend Connectivity (Optional FastAPI Service)
// ============================================================================
async function checkCloudBackend() {
  try {
    const res = await fetch(`${state.backendUrl}/api/health`, { method: "GET" });
    if (res.ok) {
      state.backendConnected = true;
      dom.backendDot.className = "pill-dot green";
      dom.backendStatusText.textContent = "Cloud Sync Active (FastAPI)";
      logEvent("INFO", "CLOUD_SYNC", "Connected to local/remote FastAPI proctoring backend", false);
    } else {
      setOfflineMode();
    }
  } catch (e) {
    setOfflineMode();
  }
}

function setOfflineMode() {
  state.backendConnected = false;
  dom.backendDot.className = "pill-dot yellow";
  dom.backendStatusText.textContent = "Client-Side Standalone (100% Browser)";
}

async function syncSessionStartToCloud() {
  try {
    await fetch(`${state.backendUrl}/api/session/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        sessionId: state.sessionId,
        candidate: state.candidate,
        startTime: new Date().toISOString()
      })
    });
  } catch (err) {
    console.warn("Backend session sync error:", err);
  }
}

async function syncEventToCloud(eventData) {
  try {
    await fetch(`${state.backendUrl}/api/session/event`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(eventData)
    });
  } catch (err) {
    console.warn("Backend event sync error:", err);
  }
}

// ============================================================================
// Event Listeners Binding
// ============================================================================
function bindEventListeners() {
  // Session Controls
  dom.btnStartSession.addEventListener("click", startProctoredSession);
  dom.btnEndSession.addEventListener("click", endProctoredSession);
  dom.btnCalibrate.addEventListener("click", calibrateNeutralPosture);
  dom.btnCaptureBackgroundMask.addEventListener("click", startRoomMaskCapture);

  // Overlay Toggles
  dom.chkShowMesh.addEventListener("change", (e) => { state.config.showMesh = e.target.checked; });
  dom.chkShowBoxes.addEventListener("change", (e) => { state.config.showBoxes = e.target.checked; });
  dom.chkShowROI.addEventListener("change", (e) => { state.config.showROI = e.target.checked; });
  dom.chkShowSuppressed.addEventListener("change", (e) => { state.config.showSuppressed = e.target.checked; });
  dom.chkShowHUD.addEventListener("change", (e) => { state.config.showHUD = e.target.checked; });

  // Feed Filter Buttons
  const filterBtns = document.querySelectorAll(".filter-btn");
  filterBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      filterBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      filterIncidentLog(btn.dataset.filter);
    });
  });

  // Export Toolbar
  dom.btnExportCSV.addEventListener("click", exportCSVAuditLog);
  dom.btnExportPDF.addEventListener("click", generatePrintableReport);

  // Settings Drawer Controls
  dom.btnSettingsOpen.addEventListener("click", openSettingsDrawer);
  dom.btnSettingsClose.addEventListener("click", closeSettingsDrawer);
  dom.settingsBackdrop.addEventListener("click", closeSettingsDrawer);
  dom.btnSettingsSave.addEventListener("click", saveSettingsDrawer);
  dom.btnSettingsReset.addEventListener("click", resetSettingsDrawer);
  dom.btnClearRoomMask.addEventListener("click", clearRoomMask);

  // Slider Readout Updates
  dom.cfgLookAwayTime.addEventListener("input", (e) => { dom.cfgLookAwayTimeVal.textContent = `${e.target.value}s`; });
  dom.cfgAbsenceTime.addEventListener("input", (e) => { dom.cfgAbsenceTimeVal.textContent = `${e.target.value}s`; });
  dom.cfgYawThreshold.addEventListener("input", (e) => { dom.cfgYawThresholdVal.textContent = `${e.target.value}°`; });
  dom.cfgPitchThreshold.addEventListener("input", (e) => { dom.cfgPitchThresholdVal.textContent = `${e.target.value}°`; });
  dom.cfgPhoneConf.addEventListener("input", (e) => { dom.cfgPhoneConfVal.textContent = `${e.target.value}%`; });
  dom.cfgBookConf.addEventListener("input", (e) => { dom.cfgBookConfVal.textContent = `${e.target.value}%`; });
  dom.cfgSkinRatio.addEventListener("input", (e) => { dom.cfgSkinRatioVal.textContent = `${e.target.value}%`; });
  dom.cfgTemporalConfirm.addEventListener("input", (e) => { dom.cfgTemporalConfirmVal.textContent = `${e.target.value} of 5 frames`; });

  // Audio Toggle
  dom.btnAudioToggle.addEventListener("click", () => {
    state.config.audioAlerts = !state.config.audioAlerts;
    dom.cfgAudioAlerts.checked = state.config.audioAlerts;
    updateAudioIcon();
  });

  // Evidence Modal Controls
  dom.btnModalClose.addEventListener("click", closeEvidenceModal);
  dom.btnModalDone.addEventListener("click", closeEvidenceModal);
  dom.btnDownloadEvidenceImage.addEventListener("click", downloadSelectedEvidenceImage);
  dom.evidenceModalBackdrop.addEventListener("click", (e) => {
    if (e.target === dom.evidenceModalBackdrop) closeEvidenceModal();
  });

  // Anti-Cheat Tab Switching Listeners
  setupAntiCheatListeners();
}

function openSettingsDrawer() {
  dom.settingsDrawer.classList.add("active");
  dom.settingsBackdrop.classList.add("active");
}

function closeSettingsDrawer() {
  dom.settingsDrawer.classList.remove("active");
  dom.settingsBackdrop.classList.remove("active");
}

function saveSettingsDrawer() {
  state.candidate.name = dom.cfgCandidateName.value || "Jane Doe";
  state.candidate.id = dom.cfgCandidateId.value || "ST-8821";
  state.candidate.examTitle = dom.cfgExamTitle.value || "CS-401 Finals";

  state.config.lookAwayTime = parseFloat(dom.cfgLookAwayTime.value);
  state.config.absenceTime = parseFloat(dom.cfgAbsenceTime.value);
  state.config.yawThreshold = parseInt(dom.cfgYawThreshold.value, 10);
  state.config.pitchThreshold = parseInt(dom.cfgPitchThreshold.value, 10);
  state.config.phoneConf = parseInt(dom.cfgPhoneConf.value, 10) / 100;
  state.config.bookConf = parseInt(dom.cfgBookConf.value, 10) / 100;
  state.config.useSkinFilter = dom.cfgSkinFilter.checked;
  state.config.skinRatioThreshold = parseInt(dom.cfgSkinRatio.value, 10) / 100;
  state.config.useMotionLock = dom.cfgMotionLock.checked;
  state.config.useWorkspaceROI = dom.cfgWorkspaceROI.checked;
  state.config.temporalConfirmFrames = parseInt(dom.cfgTemporalConfirm.value, 10);
  state.config.audioAlerts = dom.cfgAudioAlerts.checked;
  state.config.snapshotsEnabled = dom.cfgSnapshotsEnabled.checked;
  state.config.antiCheatTab = dom.cfgAntiCheatTab.checked;

  dom.headerCandidateName.textContent = `${state.candidate.name} (ID: ${state.candidate.id})`;
  updateAudioIcon();

  closeSettingsDrawer();
  logEvent("INFO", "SETTINGS_UPDATED", "Proctoring sensitivity & 4-stage suppression thresholds updated", false);
}

function resetSettingsDrawer() {
  dom.cfgCandidateName.value = "Jane Doe";
  dom.cfgCandidateId.value = "ST-8821";
  dom.cfgExamTitle.value = "CS-401 Computer Vision & AI Finals";
  dom.cfgLookAwayTime.value = 3.0;
  dom.cfgLookAwayTimeVal.textContent = "3.0s";
  dom.cfgAbsenceTime.value = 2.0;
  dom.cfgAbsenceTimeVal.textContent = "2.0s";
  dom.cfgYawThreshold.value = 22;
  dom.cfgYawThresholdVal.textContent = "22°";
  dom.cfgPitchThreshold.value = 18;
  dom.cfgPitchThresholdVal.textContent = "18°";
  dom.cfgPhoneConf.value = 20;
  dom.cfgPhoneConfVal.textContent = "20%";
  dom.cfgBookConf.value = 45;
  dom.cfgBookConfVal.textContent = "45%";
  dom.cfgSkinFilter.checked = true;
  dom.cfgSkinRatio.value = 80;
  dom.cfgSkinRatioVal.textContent = "80%";
  dom.cfgMotionLock.checked = true;
  dom.cfgWorkspaceROI.checked = true;
  dom.cfgTemporalConfirm.value = 2;
  dom.cfgTemporalConfirmVal.textContent = "2 of 5 frames";
  dom.cfgAudioAlerts.checked = true;
  dom.cfgSnapshotsEnabled.checked = true;
  dom.cfgAntiCheatTab.checked = true;
}

function updateAudioIcon() {
  if (state.config.audioAlerts) {
    dom.audioIconOn.classList.remove("hidden");
    dom.audioIconOff.classList.add("hidden");
  } else {
    dom.audioIconOn.classList.add("hidden");
    dom.audioIconOff.classList.remove("hidden");
  }
}

// Start application on page load
window.addEventListener("DOMContentLoaded", initApplication);
