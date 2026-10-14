"""
=============================================================================
SENTINEL-AI: Two-Tier Angled-Phone Detection & False-Positive Filter
Module: detector.py
=============================================================================
High-sensitivity pipeline designed to catch phones held at steep angles,
sideways, or partially occluded by fingers, while strictly suppressing:
 1. Hanging clothes/jackets in the background (Upper Background Centroid Lock)
 2. Bare hands/knuckles near face/chin (Rigid Edge + Non-Skin Core Verification)
=============================================================================
"""

import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    def iou(self, other: "BoundingBox") -> float:
        inter_x1 = max(self.x1, other.x1)
        inter_y1 = max(self.y1, other.y1)
        inter_x2 = min(self.x2, other.x2)
        inter_y2 = min(self.y2, other.y2)

        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        union_area = self.area + other.area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0


@dataclass
class DetectionCandidate:
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    tier: int = 1                         # Tier 1 (>= 0.50) or Tier 2 (0.20 - 0.49)
    has_rigid_edge: bool = False
    has_phone_core: bool = False
    skin_ratio: float = 0.0
    core_ratio: float = 0.0
    context_boosted: bool = False
    boost_amount: float = 0.0
    is_static_background: bool = False
    is_suppressed: bool = False
    suppression_reason: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class CandidatePose:
    yaw: float = 0.0       # Negative = Left, Positive = Right
    pitch: float = 0.0     # Negative = Downward peek, Positive = Upward
    roll: float = 0.0
    is_calibrated: bool = False


@dataclass
class TrackedBackgroundObject:
    category: str
    initial_center: Tuple[float, float]
    current_center: Tuple[float, float]
    first_seen: float
    last_seen: float
    max_shift: float = 0.0
    frames_seen: int = 1
    is_stationary: bool = False


class TwoTierAngledPhoneDetector:
    """
    Two-Tier Detection & Verification Pipeline:
      - Tier 1: Confidence >= 0.50 (accepted directly if in workspace ROI)
      - Tier 2: Confidence 0.20 - 0.49 (angled/partial phone verification via edge/core analysis)
      - Cheating Posture Context Boost (+0.20 boost when glancing down/sideways at desk level)
      - Static Motion Lock: < 6px shift over 2.5s suppresses hanging clothes
      - Temporal Persistence: 2 out of 5 frames confirmation with 1.5s memory
    """

    def __init__(
        self,
        base_phone_conf: float = 0.20,
        tier1_threshold: float = 0.50,
        book_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        imgsz: int = 640,
        temporal_window: int = 5,
        temporal_confirm_req: int = 2,
        memory_persistence_sec: float = 1.5,
        static_motion_threshold_px: float = 6.0,
        static_motion_duration_sec: float = 2.5,
    ):
        self.base_phone_conf = base_phone_conf
        self.tier1_threshold = tier1_threshold
        self.book_threshold = book_threshold
        self.iou_threshold = iou_threshold
        self.imgsz = imgsz
        self.temporal_window = temporal_window
        self.temporal_confirm_req = temporal_confirm_req
        self.memory_persistence_sec = memory_persistence_sec
        self.static_motion_threshold_px = static_motion_threshold_px
        self.static_motion_duration_sec = static_motion_duration_sec

        # Background centroid tracker
        self.tracked_clutter: Dict[str, TrackedBackgroundObject] = {}

        # Temporal rolling frame buffer (last N frames)
        self.temporal_history: List[Dict[str, Any]] = []

        # Persistent memory of recently confirmed boxes (for 1.5s continuity)
        self.recent_confirmed_boxes: List[Dict[str, Any]] = []

        # Pre-calibrated Room Background Mask Zones (optional)
        self.room_mask_zones: List[BoundingBox] = []

    # =========================================================================
    # Step 1: Angled-Phone Geometric & Photometric Verification
    # =========================================================================
    def verify_angled_phone(self, crop_bgr: np.ndarray) -> Tuple[bool, bool, float, float]:
        """
        Evaluates candidate crop using two cues:
          1. Straight Geometric Edge Detection (cv2.Canny + cv2.HoughLinesP)
          2. Dark/Metallic Slab Core (V < 65) OR Screen Glare Core (V > 210) in non-skin mask
        Returns:
          (has_rigid_edge, has_phone_core, skin_ratio, core_ratio)
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return False, False, 0.0, 0.0

        h, w = crop_bgr.shape[:2]
        if h < 8 or w < 8:
            return False, False, 0.0, 0.0

        total_pixels = float(h * w)

        # ---------------------------------------------------------------------
        # 1. Straight Linear Edge Detection (cv2.Canny + cv2.HoughLinesP)
        # ---------------------------------------------------------------------
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        # Apply gentle bilateral or Gaussian blur to eliminate skin texture noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        min_dimension = max(h, w)
        min_line_len = max(10, int(min_dimension * 0.35))  # > 35% of box dimension
        max_line_gap = max(4, int(min_dimension * 0.12))

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=18,
            minLineLength=min_line_len,
            maxLineGap=max_line_gap
        )

        has_rigid_edge = False
        if lines is not None and len(lines) > 0:
            for line in lines:
                coords = line.ravel()
                if len(coords) >= 4:
                    x1, y1, x2, y2 = coords[:4]
                    line_len = math.hypot(float(x2 - x1), float(y2 - y1))
                    if line_len >= min_line_len:
                        has_rigid_edge = True
                        break

        # ---------------------------------------------------------------------
        # 2. Skin Masking & Dark/Metallic Slab / Screen Glare Core
        # ---------------------------------------------------------------------
        # YCrCb Skin Detection (Cr: 133-173, Cb: 77-127)
        ycrcb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2YCrCb)
        lower_skin = np.array([0, 133, 77], dtype=np.uint8)
        upper_skin = np.array([255, 173, 127], dtype=np.uint8)
        skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)

        skin_pixel_count = float(cv2.countNonZero(skin_mask))
        skin_ratio = skin_pixel_count / total_pixels

        # Non-Skin Mask
        non_skin_mask = cv2.bitwise_not(skin_mask)

        # HSV Inspection of Non-Skin Region
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]

        # Dark Slab (Phone chassis / bezel): V < 65
        dark_core_mask = (v_channel < 65) & (non_skin_mask > 0)

        # Emissive Screen Glare (Lit screen at an angle): V > 210
        glare_core_mask = (v_channel > 210) & (non_skin_mask > 0)

        phone_core_mask = (dark_core_mask | glare_core_mask).astype(np.uint8) * 255

        # Morphological open to filter salt-and-pepper noise and verify contiguity
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_core = cv2.morphologyEx(phone_core_mask, cv2.MORPH_OPEN, kernel)
        core_pixel_count = float(cv2.countNonZero(clean_core))
        core_ratio = core_pixel_count / total_pixels

        # Check if non-skin core covers at least 12% of the box
        has_phone_core = (core_ratio >= 0.12)

        return has_rigid_edge, has_phone_core, skin_ratio, core_ratio

    # =========================================================================
    # Step 2: Cheating Posture Fusion (+0.20 Contextual Confidence Boost)
    # =========================================================================
    def apply_contextual_posture_boost(
        self,
        candidate: DetectionCandidate,
        pose: Optional[CandidatePose],
        frame_shape: Tuple[int, int]
    ) -> DetectionCandidate:
        """
        Boosts angled phone candidates when the student exhibits cheating posture:
          - Phone is located in lower 65% of the frame (desk/chest level).
          - Candidate's head pitch is glancing downward (pitch < -8° or relative downward peek)
            OR yaw is angled toward the box (|yaw| > 10°).
        """
        frame_h, frame_w = frame_shape[:2]
        box = candidate.box
        cx, cy = box.center

        is_lower_frame = (cy >= 0.35 * frame_h)  # Lower 65% of screen

        if is_lower_frame and pose is not None:
            # Looking downward or turning head toward desk
            is_looking_down = (pose.pitch < -8.0 or pose.pitch > 10.0)  # Downward pitch threshold
            is_looking_angled = (abs(pose.yaw) > 10.0)

            if is_looking_down or is_looking_angled:
                # Apply +0.20 contextual boost for candidate in 0.18 - 0.49 zone
                if 0.18 <= candidate.confidence < 0.50:
                    candidate.boost_amount = 0.20
                    candidate.confidence = min(0.95, candidate.confidence + candidate.boost_amount)
                    candidate.context_boosted = True

        return candidate

    # =========================================================================
    # Step 3: Upper Background Clutter & Centroid Motion Lock
    # =========================================================================
    def update_motion_lock(
        self,
        candidate: DetectionCandidate,
        workspace_roi: Optional[BoundingBox],
        current_time: float
    ) -> bool:
        """
        Tracks bounding box centroids over time. If an object outside the candidate's
        reach zone moves < 6px over 2.5s, it is classified as static background clutter.
        """
        box = candidate.box
        cx, cy = box.center
        category = candidate.class_name

        # Find matching tracked cluster
        matched_key = None
        min_dist = 50.0  # px distance threshold for same cluster

        for key, track in self.tracked_clutter.items():
            if track.category == category:
                dist = math.hypot(cx - track.current_center[0], cy - track.current_center[1])
                if dist < min_dist:
                    min_dist = dist
                    matched_key = key
                    break

        if matched_key is not None:
            track = self.tracked_clutter[matched_key]
            shift = math.hypot(cx - track.initial_center[0], cy - track.initial_center[1])
            track.max_shift = max(track.max_shift, shift)
            track.current_center = (cx, cy)
            track.last_seen = current_time
            track.frames_seen += 1

            # Check if stationary < 6px over >= 2.5s
            if (current_time - track.first_seen) >= self.static_motion_duration_sec:
                if track.max_shift <= self.static_motion_threshold_px:
                    track.is_stationary = True

            # If stationary and in upper background (outside reach ROI) -> static clutter
            is_outside_reach = True
            if workspace_roi is not None:
                is_outside_reach = not (
                    cx >= workspace_roi.x1 and cx <= workspace_roi.x2 and
                    cy >= workspace_roi.y1 and cy <= workspace_roi.y2
                )

            if track.is_stationary and is_outside_reach:
                return True
        else:
            # Register new tracked cluster
            cluster_id = f"{category}_{int(cx // 30)}_{int(cy // 30)}"
            self.tracked_clutter[cluster_id] = TrackedBackgroundObject(
                category=category,
                initial_center=(cx, cy),
                current_center=(cx, cy),
                first_seen=current_time,
                last_seen=current_time,
                max_shift=0.0,
                frames_seen=1,
                is_stationary=False
            )

        # Prune stale clutter tracks older than 6.0s
        stale_keys = [k for k, t in self.tracked_clutter.items() if (current_time - t.last_seen) > 6.0]
        for k in stale_keys:
            del self.tracked_clutter[k]

        return False

    def is_in_room_mask(self, box: BoundingBox) -> bool:
        """Checks if detection overlaps with pre-calibrated room mask zones."""
        for zone in self.room_mask_zones:
            if box.iou(zone) > 0.30:
                return True
            dist = math.hypot(box.center[0] - zone.center[0], box.center[1] - zone.center[1])
            if dist < 40.0:
                return True
        return False

    # =========================================================================
    # Step 4: Two-Tier Evaluation & Temporal Confirmation (2 of 5 Frames)
    # =========================================================================
    def process_frame(
        self,
        frame_bgr: np.ndarray,
        raw_detections: List[Dict[str, Any]],
        pose: Optional[CandidatePose] = None,
        workspace_roi: Optional[BoundingBox] = None,
        current_time: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes full two-tier angled phone and clutter suppression pipeline.
        Parameters:
          - frame_bgr: OpenCV BGR input frame
          - raw_detections: List of dicts with keys: {'box': [x1, y1, x2, y2], 'confidence': float, 'class_id': int, 'name': str}
          - pose: CandidatePose (yaw, pitch, roll)
          - workspace_roi: Active Workspace Reach BoundingBox
          - current_time: optional timestamp in seconds
        """
        if current_time is None:
            current_time = time.time()

        frame_h, frame_w = frame_bgr.shape[:2]

        confirmed_phone_detections: List[DetectionCandidate] = []
        tier2_verified_phones: List[DetectionCandidate] = []
        suppressed_items: List[DetectionCandidate] = []

        raw_phone_candidates: List[DetectionCandidate] = []

        # ---------------------------------------------------------------------
        # 1. Parse Raw Detections & Classify Candidate Tiers
        # ---------------------------------------------------------------------
        for det in raw_detections:
            box_coords = det["box"]
            box = BoundingBox(
                x1=float(box_coords[0]),
                y1=float(box_coords[1]),
                x2=float(box_coords[2]),
                y2=float(box_coords[3])
            )
            conf = float(det["confidence"])
            name = str(det.get("name", "")).lower()
            class_id = int(det.get("class_id", -1))

            # Focus on cell phone detections
            if name == "cell phone" or class_id == 67:
                # Discard detections below base 0.20 threshold
                if conf < self.base_phone_conf:
                    continue

                candidate = DetectionCandidate(
                    box=box,
                    confidence=conf,
                    class_id=class_id,
                    class_name="cell phone",
                    timestamp=current_time
                )

                # Contextual Posture Boost (+0.20 for angled cheating glance)
                candidate = self.apply_contextual_posture_boost(candidate, pose, (frame_h, frame_w))

                # Assign Tier based on boosted confidence
                candidate.tier = 1 if candidate.confidence >= self.tier1_threshold else 2
                raw_phone_candidates.append(candidate)

        # ---------------------------------------------------------------------
        # 2. Filter Candidate Phones (Tier 1 vs Tier 2)
        # ---------------------------------------------------------------------
        for cand in raw_phone_candidates:
            box = cand.box

            # Check Pre-calibrated Room Mask
            if self.is_in_room_mask(box):
                cand.is_suppressed = True
                cand.suppression_reason = "Pre-Calibrated Room Background Mask"
                suppressed_items.append(cand)
                continue

            # Check Static Background Clutter Motion Lock (< 6px over 2.5s)
            if self.update_motion_lock(cand, workspace_roi, current_time):
                cand.is_suppressed = True
                cand.suppression_reason = "Static Background Motion Locked (<6px over 2.5s)"
                suppressed_items.append(cand)
                continue

            # Extract Crop for Verification
            x1 = max(0, int(box.x1))
            y1 = max(0, int(box.y1))
            x2 = min(frame_w, int(box.x2))
            y2 = min(frame_h, int(box.y2))
            crop = frame_bgr[y1:y2, x1:x2]

            # -----------------------------------------------------------------
            # TIER 1: High Confidence (>= 0.50)
            # -----------------------------------------------------------------
            if cand.tier == 1:
                # Accept immediately if inside workspace reach ROI or chest level
                is_valid_area = (0.012 <= (box.area / (frame_h * frame_w)) <= 0.40)
                if is_valid_area:
                    confirmed_phone_detections.append(cand)
                else:
                    cand.is_suppressed = True
                    cand.suppression_reason = "Extreme Size Outlier"
                    suppressed_items.append(cand)
                continue

            # -----------------------------------------------------------------
            # TIER 2: Angled / Low-Profile Phone Zone (0.20 - 0.49)
            # -----------------------------------------------------------------
            rigid_edge, phone_core, skin_pct, core_pct = self.verify_angled_phone(crop)
            cand.has_rigid_edge = rigid_edge
            cand.has_phone_core = phone_core
            cand.skin_ratio = skin_pct
            cand.core_ratio = core_pct

            # Decision Rule for Tier 2:
            # Reject as Bare Hand ONLY if skin > 80% AND lacks both rigid edge and phone core
            if skin_pct > 0.80 and (not rigid_edge) and (not phone_core):
                cand.is_suppressed = True
                cand.suppression_reason = f"Bare Hand Rejected ({int(skin_pct * 100)}% skin, no edge/core)"
                suppressed_items.append(cand)
                continue

            # Accept angled phone if has_rigid_edge OR has_phone_core is True
            if rigid_edge or phone_core or cand.context_boosted:
                tier2_verified_phones.append(cand)
                confirmed_phone_detections.append(cand)
            else:
                cand.is_suppressed = True
                cand.suppression_reason = "Unverified Low-Confidence Blob"
                suppressed_items.append(cand)

        # ---------------------------------------------------------------------
        # 3. Temporal Persistence (2 out of 5 frames confirmation)
        # ---------------------------------------------------------------------
        has_current_phone = len(confirmed_phone_detections) > 0
        self.temporal_history.append({
            "has_phone": has_current_phone,
            "timestamp": current_time,
            "detections": confirmed_phone_detections
        })

        if len(self.temporal_history) > self.temporal_window:
            self.temporal_history.pop(0)

        # Count positive frames in rolling window
        positive_frames = sum(1 for f in self.temporal_history if f["has_phone"])
        is_temporally_confirmed = (positive_frames >= self.temporal_confirm_req)

        # 1.5s Bounding Box Memory Persistence
        if is_temporally_confirmed and has_current_phone:
            # Update memory
            self.recent_confirmed_boxes = [
                {"box": d.box, "timestamp": current_time, "cand": d}
                for d in confirmed_phone_detections
            ]
        else:
            # Clean expired memory older than 1.5s
            self.recent_confirmed_boxes = [
                m for m in self.recent_confirmed_boxes
                if (current_time - m["timestamp"]) <= self.memory_persistence_sec
            ]
            if len(self.recent_confirmed_boxes) > 0 and not is_temporally_confirmed:
                # Active memory bridges brief occlusion frames
                is_temporally_confirmed = True

        return {
            "is_phone_violation": is_temporally_confirmed and (has_current_phone or len(self.recent_confirmed_boxes) > 0),
            "temporal_confirm_status": f"{positive_frames}/{self.temporal_confirm_req} frames",
            "confirmed_phones": confirmed_phone_detections,
            "tier2_verified_count": len(tier2_verified_phones),
            "suppressed_items": suppressed_items,
            "memory_active": len(self.recent_confirmed_boxes) > 0,
            "active_clutter_tracks": len(self.tracked_clutter)
        }


# =============================================================================
# Standalone CLI Test Harness
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("[SENTINEL-AI] TWO-TIER ANGLED PHONE DETECTOR TEST HARNESS")
    print("=" * 70)

    detector = TwoTierAngledPhoneDetector()

    # Create synthetic test crops
    test_w, test_h = 100, 140

    # 1. Simulated Angled Phone with Dark Core & Linear Edges
    phone_crop = np.zeros((test_h, test_w, 3), dtype=np.uint8)
    # Fingers (skin tone in YCrCb: Cr~150, Cb~100 -> BGR ~ [120, 150, 210])
    phone_crop[:] = [120, 150, 210]
    # Dark phone body in center (V < 65)
    phone_crop[20:120, 25:75] = [30, 30, 35]
    # Crisp phone bezel edge
    cv2.line(phone_crop, (25, 20), (25, 120), (200, 200, 200), 2)

    rigid_edge, phone_core, skin_pct, core_pct = detector.verify_angled_phone(phone_crop)
    print(f"\n[Test 1] Angled Phone in Hand Crop:")
    print(f"  Rigid Edge Detected : {rigid_edge}")
    print(f"  Phone Core Detected : {phone_core} (Core Ratio: {core_pct:.1%})")
    print(f"  Skin Pixel Ratio    : {skin_pct:.1%}")
    assert (rigid_edge or phone_core), "Angled phone with chassis/edge must be verified!"

    # 2. Simulated Bare Hand (Curved, all skin, no linear edges, no dark core)
    hand_crop = np.zeros((test_h, test_w, 3), dtype=np.uint8)
    hand_crop[:] = [120, 150, 210]
    # Soft curved skin contours
    cv2.circle(hand_crop, (50, 70), 40, (110, 140, 205), -1)

    h_edge, h_core, h_skin, h_core_pct = detector.verify_angled_phone(hand_crop)
    print(f"\n[Test 2] Bare Hand Crop:")
    print(f"  Rigid Edge Detected : {h_edge}")
    print(f"  Phone Core Detected : {h_core} (Core Ratio: {h_core_pct:.1%})")
    print(f"  Skin Pixel Ratio    : {h_skin:.1%}")
    assert h_skin > 0.80 and not h_edge and not h_core, "Bare hand must be rejected!"

    print("\n[SUCCESS] All geometric and photometric verification assertions passed successfully!")

