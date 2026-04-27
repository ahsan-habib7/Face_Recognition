# SecureBank KYC System — Complete Architecture & Deployment Guide

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BROWSER (User's device)                          │
│                                                                       │
│  Index.cshtml                                                         │
│  ┌────────────────┐   ┌────────────────────────────────────────┐     │
│  │  ID Card       │   │  Camera Modal (Liveness Check)         │     │
│  │  upload zone   │   │                                         │     │
│  │                │   │  Stage 1: ALIGN  — frontal face hold   │     │
│  │  [Drag/click   │   │  Stage 2: BLINK  — EAR x3 detections  │     │
│  │   to upload]   │   │  Stage 3: TURN   — L→R→L head turn    │     │
│  │                │   │  Stage 4: MOTION — ring-buffer vote    │     │
│  │                │   │  Stage 5: CAPTURE — auto selfie        │     │
│  └────────────────┘   └────────────────────────────────────────┘     │
│                                    │ AJAX frames @ 120ms              │
│                            ┌───────▼───────┐                         │
│                            │  Form submit  │                         │
│                            │  POST Verify  │                         │
│                            │  (multipart)  │                         │
│                            └───────┬───────┘                         │
└────────────────────────────────────│────────────────────────────────-┘
                                     │ HTTP POST (same origin)
┌────────────────────────────────────▼────────────────────────────────┐
│              ASP.NET Core MVC (.NET 8)  — Port 5000                  │
│                                                                       │
│  VerificationController                                               │
│  ┌─────────────────────────────────────────────────────────┐         │
│  │  GET  /Verification/Index     → return View()            │         │
│  │  POST /Verification/ValidateId → AJAX → JSON            │         │
│  │                                                           │         │
│  │  POST /Verification/LivenessStart  ─────────────────────┼──┐      │
│  │  POST /Verification/LivenessFrame  ─ proxy to Python ───┼──┤      │
│  │  POST /Verification/LivenessCancel ─────────────────────┼──┘      │
│  │                                                           │         │
│  │  POST /Verification/Verify                               │         │
│  │    1. Guard: idImage null? → return View("Result", err) │         │
│  │    2. Guard: sessionId empty? → return View("Result", err│         │
│  │    3. Call Python /verify-with-liveness?session_id=xxx  │         │
│  │    4. Map JSON → FaceVerificationResult ViewModel        │         │
│  │    5. Compare userName/userNid/userDob with NID OCR data │         │
│  │    6. return View("Result", model)  ◄── ALWAYS          │         │
│  └─────────────────────────────────────────────────────────┘         │
│                                                                       │
│  FaceVerificationService  (typed HttpClient)                          │
│    ValidateIdAsync()       → POST /validate-id                        │
│    VerifyWithLivenessAsync() → POST /verify-with-liveness             │
└─────────────────────────────┬────────────────────────────────────────┘
                              │ Internal Docker network
                              │ http://face-api:8000
┌─────────────────────────────▼────────────────────────────────────────┐
│              Python FastAPI  — Port 8000                              │
│                                                                       │
│  POST /validate-id                                                    │
│    1. Decode uploaded image                                           │
│    2. Upscale → perspective correction → denoise → threshold          │
│    3. Tesseract OCR (eng+ben)                                         │
│    4. Keyword match: NID / Passport / Driving License                 │
│    5. Return { valid, id_type, message }                              │
│                                                                       │
│  POST /liveness/start                                                 │
│    1. Create session (UUID)                                           │
│    2. Return { session_id, blink_needed: 3 }                          │
│                                                                       │
│  POST /liveness/frame?session_id=xxx                                  │
│    Stage: align  → InsightFace 5-pt yaw estimate, hold 8 frames      │
│    Stage: blink  → MediaPipe 478-pt EAR state machine, 3 blinks      │
│    Stage: turn   → nose/eye_mid ratio L→R→L, hold 5 frames each     │
│    Stage: motion → 5-frame ring buffer, 3/5 above threshold          │
│    Stage: done   → auto-capture selfie JPEG → store in session       │
│                                                                       │
│  POST /verify-with-liveness?session_id=xxx                           │
│    1. Get session, check liveness_passed                             │
│    2. Read auto_selfie JPEG from session                             │
│    3. InsightFace ArcFace: detect → align → embed (512-dim)         │
│       on both ID card face AND selfie face                           │
│    4. Cosine similarity of L2-normed embeddings                      │
│    5. Threshold: 0.35 (Medium+), 0.55 (High — Verified)             │
│    6. Tesseract OCR: extract name, NID no, DOB, address             │
│    7. Return full result with base64 face crops + NID data          │
│    8. Delete session                                                  │
│                                                                       │
│  Models loaded at startup:                                            │
│    buffalo_l/det_10g.onnx    → RetinaFace detector                   │
│    buffalo_l/2d106det.onnx   → 106-pt landmark detector              │
│    buffalo_l/w600k_r50.onnx  → ArcFace ResNet-50 embedder           │
│    face_landmarker.task       → MediaPipe 478-pt mesh (blink EAR)   │
└──────────────────────────────────────────────────────────────────────┘

Result.cshtml (rendered server-side by ASP.NET)
┌──────────────────────────────────────────────────────────────────────┐
│  LEFT PANEL                    │  RIGHT PANEL                        │
│  ─────────────────────         │  ─────────────────────              │
│  ✓ ID Card Uploaded            │  [Result icon: ✅/⚠️/❌]           │
│  ✓ Liveness Verified           │  Status badge                       │
│  ③ AI Verification Result      │  Result title                       │
│                                │  Result message                     │
│  [NID Face] | [Live Selfie]   │                                     │
│                                │  ████████████░░░░  72.4%           │
│  📋 Extracted NID Data         │  Similarity meter                   │
│  Name:    Mohammad Ali  ✅     │  Confidence: High                   │
│  NID No:  1234567890    ✅     │                                     │
│  DOB:     01 Jan 1990   ✅     │  [Login Auth]  [KYC Status]        │
│  Address: Dhaka, BD            │  [Acct Opening] [Txn Auth]         │
│                                │                                     │
│  👤 Your Input (if supplied)   │  [← Try Again] [🏠 Home]          │
│  Name:  Mohammad Ali    ✅     │                                     │
│  NID:   1234567890      ✅     │                                     │
│  DOB:   01 Jan 1990     ✅     │                                     │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Bug Fix Log

### Bug 1 — `Result.cshtml` never reached (CRITICAL)
**Root cause:** `return View("Result", model)` was only reached if no exceptions
occurred AND no early returns fired. Multiple code paths returned `View("Index")`
or threw before this line.

**Fix:** Every single code path in `Verify()` now ends with `return View("Result", model)`.
Even errors set `model.Error` and return the Result view.

### Bug 2 — `sessionId` hidden field empty on submit
**Root cause:** JS populated `#livenessSessionId` as the LAST step in the `done`
handler, after canvas/preview work that could throw and halt execution.

**Fix:** `hf.value = sessionId` is now the FIRST statement inside the `done`
block, before any other work. If canvas preview fails, session ID is already saved.

### Bug 3 — Double `/liveness/` prefix → 404 → session never created
**Root cause:** Router registered routes as `/liveness/start` AND `main.py`
mounted with `prefix="/liveness"` → final URL `/liveness/liveness/start` → 404.
`LivenessStart` proxy got 404 → threw → `openCamera()` caught error → `sessionId`
was never set.

**Fix:** Router routes are `/start`, `/frame`, `/cancel` (no prefix). `main.py`
adds the `/liveness` prefix via `include_router(prefix="/liveness")`.

### Bug 4 — Head-turn challenge physically impossible to trigger
**Root cause:** Measured `nose_x - bbox_centre_x`. When the head turns, the
entire bounding box shifts in the same direction as the nose, so the difference
stays near zero regardless of head angle.

**Fix:** Measure `(nose_x - eye_midpoint_x) / eye_distance`. The eye midpoint
is a stable anchor — the nose moves significantly relative to it during head rotation.
- Frontal:    ratio ≈  0.00
- 15° left:   ratio ≈ -0.20
- 15° right:  ratio ≈ +0.20

### Bug 5 — `FRAME_MS = 12` (83fps → server overload)
**Root cause:** Typo. Comment said 120ms but code used 12ms.

**Fix:** `const FRAME_MS = 120` — 8fps, safely above Python's `MIN_FRAME_MS = 80ms`.

### Bug 6 — `"PythonApi"` named client had no BaseAddress
**Root cause:** Only the typed `FaceVerificationService` client had `BaseAddress`.
`VerificationController` used `IHttpClientFactory.CreateClient("PythonApi")` but
this named client was not registered → bare `HttpClient` with no base URL →
all liveness proxy calls failed with connection errors.

**Fix:** Registered in `Program.cs`:
```csharp
builder.Services.AddHttpClient("PythonApi", client => {
    client.BaseAddress = new Uri(apiUrl);
    client.Timeout = TimeSpan.FromSeconds(30);
});
```

---

## Deployment

### Prerequisites
- Docker Desktop (Windows/Mac) or Docker Engine (Linux)
- 8GB RAM minimum (ArcFace buffalo_l models are ~300MB)
- Webcam for liveness detection

### Quick Start

```bash
# Clone / extract project
cd RecognizeUser

# Build and start both containers
docker-compose up --build

# First run takes 3-5 minutes (downloading ArcFace models)
# Subsequent runs: ~30 seconds
```

Open browser: http://localhost:5000

### Docker container logs
```bash
# Python backend logs (face detection, liveness stages, OCR)
docker logs -f face_api

# .NET frontend logs
docker logs -f face_frontend
```

### Local development (without Docker)

**Python backend:**
```bash
cd InsightFace_BackendAPI
pip install -r requirements.txt
# Download face_landmarker.task from MediaPipe model garden
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

**ASP.NET frontend:**
```bash
cd FaceRecognition_Frontend/FaceRecognition_Frontend
# appsettings.json already has FaceApiBaseUrl: http://localhost:8000
dotnet run
```

---

## Execution Flow (step by step)

1. **User opens** http://localhost:5000
   → `GET /Verification/Index` → `return View()`

2. **User uploads ID card image**
   → JS calls `AJAX POST /Verification/ValidateId` (with CSRF token)
   → Controller calls `FaceVerificationService.ValidateIdAsync()`
   → HTTP POST to Python `/validate-id`
   → Tesseract OCR: `preprocess_for_ocr()` → `extract_text()` → `detect_id_type()`
   → Returns `{ valid: true, id_type: "Bangladesh NID" }`
   → JS: marks Step 1 done ✓, activates Step 2

3. **User clicks "Open Camera"**
   → `openCamera()` calls `navigator.mediaDevices.getUserMedia()`
   → `POST /Verification/LivenessStart` (ASP.NET proxy)
   → Python: `create_session()` → returns `{ session_id: "uuid" }`
   → JS: stores `sessionId`, starts frame send loop at 120ms interval

4. **Liveness frames sent** (each frame):
   → `sendFrame()` captures 640×480 JPEG blob (unmirrored)
   → `POST /Verification/LivenessFrame?sessionId=xxx`
   → ASP.NET proxy → Python `/liveness/frame?session_id=xxx`
   → `process_frame_logic()` runs current challenge logic
   → Returns `{ stage, progress, icon, message, ... }`
   → JS: updates UI (oval, pills, instruction text, blink dots, turn boxes)

5. **All 3 challenges pass** (blink + turn + motion):
   → Python: captures selfie JPEG into `session["auto_selfie"]`
   → Returns `{ stage: "done", passed: true }`
   → **JS FIRST:** `hf.value = sessionId` (hidden field populated)
   → JS: shows selfie preview, marks Step 2 done ✓, auto-closes modal

6. **User clicks "Verify My Identity"**
   → `validateForm()` confirms both `idCardValid` and `selfieReady` and `sessionId`
   → Standard `<form>` POST to `POST /Verification/Verify` (multipart/form-data)
   → Contains: `idImage` file + `sessionId` hidden field

7. **Controller processes verification:**
   → Guard checks: `idImage != null`, `sessionId != empty`
   → `FaceVerificationService.VerifyWithLivenessAsync(idImage, sessionId)`
   → `POST /verify-with-liveness?session_id=xxx` with ID card image
   → Python: reads `session["auto_selfie"]`, runs ArcFace on both images
   → Python: extracts OCR data from ID card
   → Python: deletes session, returns full JSON result
   → Controller: deserializes → `FaceVerificationResult` ViewModel
   → Controller: compares `userName/userNid/userDob` with NID OCR data
   → **`return View("Result", model)`** ← THIS IS THE FIX

8. **Result.cshtml renders:**
   → Left: Steps indicator + NID face / Selfie face + OCR data table
   → Right: Result card (Verified/Review/Failed) + match % meter + KYC grid

---

## Security Notes (Bank-Grade)

- **Fully offline** — no external APIs, all models run locally
- **No permanent storage** — images are processed in memory and discarded
- **Liveness anti-spoof** — 4-layer protection:
  1. Laplacian texture check (rejects printed photos)
  2. EAR blink state machine (rejects static images)
  3. Head-turn sequence (rejects 2D spoofs)
  4. Motion ring-buffer voting (rejects screens)
- **Session expiry** — liveness sessions auto-expire after 10 minutes
- **CSRF protection** — all form POSTs require `__RequestVerificationToken`
- **Input sanitization** — NID number comparison strips spaces/dashes
- **Error isolation** — exceptions caught; error shown in Result view, not leaked
