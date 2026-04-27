import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial import distance as dist
from insightface.app import FaceAnalysis


# InsightFace — face detection + embeddings
_iface = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
_iface.prepare(ctx_id=0, det_size=(640, 640))

# MediaPipe 478-point face mesh — blink EAR 
MODEL_PATH  = "/app/face_landmarker.task"
_mp_options = mp_vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
_face_mesh = mp_vision.FaceLandmarker.create_from_options(_mp_options)

# MediaPipe eye landmark indices
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

ALIGN_HOLD_FRAMES = 8
YAW_MAX           = 14.0
PITCH_MAX         = 12.0
EAR_CLOSE         = 0.21
EAR_OPEN          = 0.27
EAR_CLOSE_NEED    = 2
EAR_OPEN_NEED     = 3
TEXTURE_MIN       = 60.0
TURN_THRESH       = 0.18
TURN_HOLD_FRAMES  = 5
MOTION_THRESH     = 3.0
MOTION_NEED       = 2
MIN_FRAME_MS      = 80

CONSISTENCY_THRESHOLD_FRONTAL = 0.30  
CONSISTENCY_THRESHOLD_PROFILE = 0.22  

def _ear(landmarks, indices: list) -> float:
    pts = [(landmarks[i].x, landmarks[i].y) for i in indices]
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-6)


def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _yaw_deg(face) -> float | None:
    if face.kps is None:
        return None
    kps    = face.kps
    le, re = kps[0], kps[1]
    nose   = kps[2]
    eye_cx = (le[0] + re[0]) / 2.0
    eye_w  = abs(re[0] - le[0])
    if eye_w < 1:
        return None
    return float((nose[0] - eye_cx) / eye_w * 90.0)

def _pitch_deg(face) -> float | None:
    if face.kps is None:
        return None

    kps = face.kps
    le, re = kps[0], kps[1]
    nose   = kps[2]

    eye_cy = (le[1] + re[1]) / 2.0
    eye_h  = abs(re[0] - le[0])  

    if eye_h < 1:
        return None

    # Positive → looking down, Negative → looking up
    return float((nose[1] - eye_cy) / eye_h * 90.0)


def _encode_jpeg(img: np.ndarray, quality: int = 94) -> bytes | None:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok and buf is not None:
        return buf.tobytes()
    ok2, buf2 = cv2.imencode(".jpg", img)
    return buf2.tobytes() if (ok2 and buf2 is not None) else None


def _get_embedding(image_bgr: np.ndarray) -> np.ndarray | None:
    try:
        faces = _iface.get(image_bgr)
        if not faces:
            return None
        face = max(faces, key=lambda f: f.det_score)
        emb  = face.normed_embedding
        if emb is None:
            return None
        # Ensure L2-normalised
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 0 else emb
    except Exception as e:
        print(f"[EMBED] Error extracting embedding: {e}", flush=True)
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def _capture_and_embed(image_bgr: np.ndarray, label: str) -> tuple[bytes | None, np.ndarray | None]:
    jpeg = _encode_jpeg(image_bgr, quality=95)
    emb  = _get_embedding(image_bgr)
    if emb is not None:
        print(f"[CAPTURE] {label}: stored JPEG={len(jpeg) if jpeg else 0}B  emb shape={emb.shape}", flush=True)
    else:
        print(f"[CAPTURE] {label}: WARNING — no face detected in capture frame", flush=True)
    return jpeg, emb


def _run_consistency_check(session: dict) -> dict:
    emb1 = session.get("capture_1_emb")
    emb2 = session.get("capture_2_emb")
    emb3 = session.get("capture_3_emb")

    scores = {}
    failures = []

    pairs = [
        ("1v2", emb1, emb2, CONSISTENCY_THRESHOLD_PROFILE),   # frontal vs profile
        ("1v3", emb1, emb3, CONSISTENCY_THRESHOLD_FRONTAL),   # frontal vs frontal
        ("2v3", emb2, emb3, CONSISTENCY_THRESHOLD_PROFILE),   # profile vs frontal
    ]

    for label, ea, eb, threshold in pairs:
        if ea is None or eb is None:
            scores[label] = 0.0
            failures.append(f"{label}:no_face")
            print(f"[CONSISTENCY] {label}: FAIL — embedding missing", flush=True)
        else:
            sim = _cosine_similarity(ea, eb)
            scores[label] = round(sim, 4)
            if sim < threshold:
                failures.append(f"{label}:{sim:.3f}")
                print(f"[CONSISTENCY] {label}: FAIL sim={sim:.3f} < threshold={threshold}", flush=True)
            else:
                print(f"[CONSISTENCY] {label}: PASS sim={sim:.3f}", flush=True)

    ok = len(failures) == 0
    session["consistency_ok"]     = ok
    session["consistency_scores"] = scores
    session["spoof_flagged"]       = not ok

    return {
        "consistency_ok":     ok,
        "consistency_scores": scores,
        "spoof_flagged":      not ok,
        "failures":           failures,
    }

def process_frame_logic(session: dict, image: np.ndarray) -> dict:
    now = time.time()
    if (now - session["_last_ts"]) * 1000 < MIN_FRAME_MS:
        return _state_response(session)
    session["_last_ts"] = now

    stage = session["stage"]

    # STAGE: align
    if stage == "align":
        faces = _iface.get(image)
        if not faces:
            session["_align_ok_frames"] = 0
            return {
                "stage": "align", "progress": 0,
                "icon": "👤", "message": "No face detected — position your face in the oval"
            }

        face = max(faces, key=lambda f: f.det_score)
        yaw  = _yaw_deg(face)

        if yaw is None or abs(yaw) > YAW_MAX:
            session["_align_ok_frames"] = 0
            side = "right" if (yaw or 0) > 0 else "left"
            return {
                "stage": "align", "progress": 5,
                "icon": "↔️", "message": f"Face angled {side} — look straight at the camera"
            }

        session["_align_ok_frames"] += 1
        hold   = session["_align_ok_frames"]
        needed = ALIGN_HOLD_FRAMES

        if hold >= needed:
            session["stage"] = "blink"
            return {
                "stage": "blink", "challenge_done": "align",
                "progress": 5, "icon": "✅",
                "message": "Face aligned! Now blink 3 times naturally",
                "blink_count": 0, "blink_needed": 3
            }

        return {
            "stage": "align",
            "progress": int(hold / needed * 10),
            "icon": "🎯",
            "message": f"Hold still… ({hold}/{needed})"
        }

    # STAGE: blink
    if stage == "blink":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if _lap_var(gray) < TEXTURE_MIN:
            return {
                "stage": "blink",
                "blink_count": session["blink_count"], "blink_needed": 3,
                "progress": int(session["blink_count"] / 3 * 33),
                "icon": "⚠️", "message": "Image too flat — please use your real face (not a photo)"
            }

        rgb     = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = _face_mesh.detect(mp_img)

        if not results.face_landmarks:
            return {
                "stage": "blink",
                "blink_count": session["blink_count"], "blink_needed": 3,
                "progress": int(session["blink_count"] / 3 * 33),
                "icon": "👁️", "message": "No face detected — look directly at the camera"
            }

        for lm in results.face_landmarks:
            ear = (_ear(lm, LEFT_EYE) + _ear(lm, RIGHT_EYE)) / 2.0

            state     = session["_eye_state"]
            close_ctr = session["_close_ctr"]
            open_ctr  = session["_open_ctr"]

            if state in ("open", "closing"):
                if ear < EAR_CLOSE:
                    close_ctr += 1
                    state = "closing"
                    if close_ctr >= EAR_CLOSE_NEED:
                        state     = "closed"
                        open_ctr  = 0
                else:
                    close_ctr = 0
                    state     = "open"
            elif state == "closed":
                if ear > EAR_OPEN:
                    open_ctr += 1
                    if open_ctr >= EAR_OPEN_NEED:
                        session["blink_count"] += 1
                        state     = "open"
                        close_ctr = 0
                        open_ctr  = 0
                        print(f"[BLINK] count={session['blink_count']} EAR={ear:.3f}", flush=True)

            session["_eye_state"] = state
            session["_close_ctr"] = close_ctr
            session["_open_ctr"]  = open_ctr

            count = session["blink_count"]
            if count >= 3:
                # CAPTURE 1 — blink challenge complete
                print("[CAPTURE 1] Blink stage complete — capturing frame 1", flush=True)
                jpeg1, emb1 = _capture_and_embed(image, "capture_1")
                session["capture_1"]     = jpeg1
                session["capture_1_emb"] = emb1

                session["stage"] = "turn"
                return {
                    "stage": "turn", "challenge_done": "blink",
                    "progress": 33, "icon": "✅",
                    "message": "3 blinks confirmed! Turn head: RIGHT → LEFT → RIGHT",
                    "blink_count": 3, "blink_needed": 3
                }

            return {
                "stage": "blink", "blink_count": count, "blink_needed": 3,
                "progress": int(count / 3 * 33),
                "icon": "👁️", "message": f"Blink naturally ({count}/3 detected)"
            }

        return {
            "stage": "blink", "blink_count": session["blink_count"], "blink_needed": 3,
            "progress": int(session["blink_count"] / 3 * 33),
            "icon": "👁️", "message": "Blink naturally"
        }

    # STAGE: turn
    if stage == "turn":
        faces = _iface.get(image)
        seq   = session["turn_sequence"]
        base_progress = int(33 + len(seq) / 3 * 33)

        if not faces:
            return {
                "stage": "turn", "turn_sequence": seq,
                "current_direction": "none",
                "progress": base_progress, "icon": "↔️",
                "message": "No face detected — keep your face in frame"
            }

        face = max(faces, key=lambda f: f.det_score)

        if face.kps is None:
            return {
                "stage": "turn", "turn_sequence": seq,
                "current_direction": "none",
                "progress": base_progress, "icon": "↔️",
                "message": "Keypoints unavailable — move closer to the camera"
            }

        kps = face.kps
        eye_mid_x = (kps[0][0] + kps[1][0]) / 2.0
        eye_dist  = abs(kps[1][0] - kps[0][0])

        if eye_dist < 10:
            return {
                "stage": "turn", "turn_sequence": seq,
                "current_direction": "none",
                "progress": base_progress, "icon": "↔️",
                "message": "Move closer to the camera"
            }

        yaw_ratio = (kps[2][0] - eye_mid_x) / eye_dist

        if   yaw_ratio < -TURN_THRESH: direction = "left"
        elif yaw_ratio >  TURN_THRESH: direction = "right"
        else:                          direction = "center"

        if direction == session["_turn_dir"]:
            session["_turn_hold"] += 1
        else:
            session["_turn_dir"]  = direction
            session["_turn_hold"] = 1

        hold = session["_turn_hold"]

        if hold >= TURN_HOLD_FRAMES:
            if   len(seq) == 0 and direction == "left":
                seq.append("left");  session["_turn_hold"] = 0
            elif len(seq) == 1 and direction == "right":
                seq.append("right"); session["_turn_hold"] = 0
            elif len(seq) == 2 and direction == "left":
                seq.append("left");  session["_turn_hold"] = 0

        print(
            f"[TURN] yaw={yaw_ratio:+.3f}  dir={direction}  "
            f"hold={hold}/{TURN_HOLD_FRAMES}  seq={seq}",
            flush=True
        )

        if len(seq) >= 3:
            print("[CAPTURE 2] Turn stage complete — capturing frame 2", flush=True)
            jpeg2, emb2 = _capture_and_embed(image, "capture_2")
            session["capture_2"]     = jpeg2
            session["capture_2_emb"] = emb2

            session["stage"] = "motion"
            return {
                "stage": "motion", "challenge_done": "turn",
                "progress": 66, "icon": "✅",
                "message": "Head turns confirmed! Move naturally for motion check"
            }

        msgs = ["Turn head RIGHT ▶", "Now turn LEFT ◀", "Back to RIGHT ▶"]
        return {
            "stage":             "turn",
            "turn_sequence":     seq,
            "current_direction": direction,
            "yaw_ratio":         round(yaw_ratio, 3),
            "hold_count":        hold,
            "hold_needed":       TURN_HOLD_FRAMES,
            "progress":          int(33 + len(seq) / 3 * 33),
            "icon":              "↔️",
            "message":           msgs[len(seq)],
        }

    # STAGE: motion
    if stage == "motion":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        prev = session.get("_prev_gray")

        diff = float(np.mean(cv2.absdiff(prev, gray))) if prev is not None else 0.0
        session["_prev_gray"] = gray

        buf = session["_motion_buf"]
        buf.append(diff)
        if len(buf) > 5:
            buf.pop(0)
        session["_motion_buf"] = buf

        active = sum(1 for d in buf if d > MOTION_THRESH)
        print(f"[MOTION] diff={diff:.2f} active={active}/5", flush=True)

        if session["_motion_passed"]:
            faces = _iface.get(image)
            if not faces:
                return {
                    "stage": "motion",
                    "progress": 92,
                    "icon": "👤",
                    "message": "Please look straight at the camera"
                }
                
            face = max(faces, key=lambda f: f.det_score)
            yaw  = _yaw_deg(face)

            if yaw is not None and abs(yaw) > YAW_MAX:
                side = "right" if (yaw or 0) > 0 else "left"
                return {
                    "stage": "motion",
                    "progress": 92,
                    "icon": "↔️",
                    "message": f"Face angled {side} — look straight at the camera"
                }

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if _lap_var(gray) < TEXTURE_MIN:
                return {
                    "stage": "motion",
                    "progress": 92,
                    "icon": "⚠️",
                    "message": "Image unclear — hold phone steady"
                }
            # Still frame — Capture 3 here
            print("[CAPTURE 3] Motion stage complete — capturing final frame 3", flush=True)
            jpeg3, emb3 = _capture_and_embed(image, "capture_3")
            session["capture_3"]     = jpeg3
            session["capture_3_emb"] = emb3
            session["auto_selfie"]   = jpeg3   # legacy key

            consistency = _run_consistency_check(session)

            if consistency["consistency_ok"]:
                session["liveness_passed"] = True
                session["stage"]           = "done"
                print(
                    f"[SECURITY] Consistency PASS — scores: {consistency['consistency_scores']}",
                    flush=True
                )
                return {
                    "stage":              "done",
                    "passed":             True,
                    "auto_captured":      True,
                    "progress":           100,
                    "icon":               "✅",
                    "message":            "Liveness confirmed! Identity ready to verify.",
                    "consistency_ok":     True,
                    "consistency_scores": consistency["consistency_scores"],
                }
            else:
                session["liveness_passed"] = False
                session["stage"]           = "spoof"
                print(
                    f"[SECURITY] ⚠️  SPOOF DETECTED — consistency FAIL "
                    f"scores={consistency['consistency_scores']} "
                    f"failures={consistency['failures']}",
                    flush=True
                )
                return {
                    "stage":              "spoof",
                    "passed":             False,
                    "auto_captured":      True,
                    "progress":           100,
                    "icon":               "🚨",
                    "message":            "⚠️ Verification failed: different faces detected during liveness check. Possible identity spoofing attempt.",
                    "consistency_ok":     False,
                    "consistency_scores": consistency["consistency_scores"],
                    "spoof_flagged":      True,
                    "failures":           consistency["failures"],
                }

        if active >= MOTION_NEED:
            session["_motion_passed"] = True
            return {
                "stage": "motion", "passed": False,
                "progress": 92, "icon": "📸",
                "message": "Hold still for capture…"
            }

        return {
            "stage": "motion", "passed": False,
            "progress": int(66 + min(active / MOTION_NEED, 1.0) * 24),
            "icon": "🔍",
            "message": "Detecting live motion… breathe or nod slightly"
        }

    if stage == "done":
        return {
            "stage":          "done",
            "passed":         session.get("liveness_passed", False),
            "auto_captured":  session.get("auto_selfie") is not None,
            "progress":       100,
            "icon":           "✅",
            "message":        "Liveness confirmed! Identity ready to verify.",
            "consistency_ok": session.get("consistency_ok"),
        }

    # STAGE: spoof — consistency check failed
    if stage == "spoof":
        return {
            "stage":         "spoof",
            "passed":        False,
            "progress":      100,
            "icon":          "🚨",
            "message":       "Verification failed: identity spoofing detected. Please restart.",
            "spoof_flagged": True,
        }

    return {"stage": "done", "passed": True, "auto_captured": True}


def _state_response(session: dict) -> dict:
    s = session.get("stage", "align")

    if s == "align":
        h = session.get("_align_ok_frames", 0)
        return {
            "stage": "align",
            "progress": int(h / ALIGN_HOLD_FRAMES * 10),
            "icon": "🎯", "message": "Position your face in the oval"
        }
    if s == "blink":
        c = session.get("blink_count", 0)
        return {
            "stage": "blink", "blink_count": c, "blink_needed": 3,
            "progress": int(c / 3 * 33), "icon": "👁️",
            "message": f"Blink naturally ({c}/3)"
        }
    if s == "turn":
        seq = session.get("turn_sequence", [])
        return {
            "stage": "turn", "turn_sequence": seq,
            "progress": int(33 + len(seq) / 3 * 33),
            "icon": "↔️", "message": "Turn head L→R→L"
        }
    if s == "motion":
        return {
            "stage": "motion", "passed": False,
            "progress": 70, "icon": "🔍", "message": "Detecting motion…"
        }
    if s == "spoof":
        return {
            "stage": "spoof", "passed": False,
            "progress": 100, "icon": "🚨",
            "message": "Spoofing detected — please restart"
        }
    return {
        "stage": "done", "passed": True, "auto_captured": True,
        "progress": 100, "icon": "✅", "message": "Done"
    }