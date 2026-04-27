"""
SECURITY ENHANCEMENT: Multi-Stage Image Capture & Consistency Verification
---------------------------------------------------------------------------
    Three selfies are captured at distinct challenge boundaries:
      capture_1  — taken immediately when blink challenge completes (blink→turn transition)
      capture_2  — taken when head-turn challenge completes (turn→motion transition)
      capture_3  — taken at the final motion capture (existing auto_selfie)
"""

import time
import uuid

sessions: dict = {}


def create_session() -> str:
    """Create a new liveness session and return its ID."""
    sid = str(uuid.uuid4())
    sessions[sid] = {
        # Pipeline stage: align → blink → turn → motion → done
        "stage": "align",

        # Stage: align
        "_align_ok_frames": 0,

        # Stage: blink
        "blink_count":  0,
        "_eye_state":   "open",
        "_close_ctr":   0,
        "_open_ctr":    0,

        # Stage: turn
        "turn_sequence": [],
        "_turn_dir":    "center",
        "_turn_hold":   0,

        # Stage: motion
        "_prev_gray":     None,
        "_motion_buf":    [],
        "_motion_passed": False,

        # MULTI-STAGE CAPTURE 
        "capture_1":     None,   
        "capture_1_emb": None,   # ArcFace 512-dim embedding

        "capture_2":     None,   # JPEG bytes at turn→motion boundary
        "capture_2_emb": None,

        "capture_3":     None,   # JPEG bytes at motion final capture
        "capture_3_emb": None,

        "consistency_ok":     None,   # None=unchecked, True=pass, False=fail
        "consistency_scores": {},     # {"1v2": float, "1v3": float, "2v3": float}
        "spoof_flagged":      False,  # True if subject switch detected

        # Legacy keys (backward compat with verify endpoint)
        "auto_selfie":     None,    # same as capture_3
        "liveness_passed": False,   # True only when consistency passes too

        "_last_ts":  0.0,
        "last_seen": time.time(),
    }
    return sid


def get_session(sid: str) -> dict | None:
    s = sessions.get(sid)
    if s:
        s["last_seen"] = time.time()
    return s


def delete_session(sid: str) -> None:
    sessions.pop(sid, None)


def cleanup_old_sessions() -> int:
    now   = time.time()
    stale = [sid for sid, s in sessions.items() if now - s["last_seen"] > 600]
    for sid in stale:
        sessions.pop(sid, None)
    return len(stale)
